from comet_ml import Experiment
from tqdm import tqdm
import torch
from fastai.vision import *
### added by Wenjie (Oct 19)
import warnings
warnings.filterwarnings("ignore")
###
from image_classification.utils.utils import *

class KDTrainer:

    def __init__(self, student, teacher, data, sf_teacher, sf_student, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, pruning=False, bits_weight_sharing=0, expt=None):
        self.student = student 
        self.teacher =  teacher
        self.data = data
        # we will not use sf_xx
        self.sf_teacher = sf_teacher
        self.sf_student = sf_student

        self.loss_function = loss_function
        self.loss_function2 = loss_function2
        self.optimizer = optimizer
        self.hyper_params = hyper_params
        self.num_epoch = epoch
        self.savename = savename
        self.best_val_acc = best_val_acc
        self.pruning = pruning
        self.bits_weight_sharing = bits_weight_sharing
        self.expt = expt

    def train(self, quartized=False, gpu=0):
        # to keep consistent with the original train  method
        print(f'train gpu:{gpu}')
        max_val_acc = self.best_val_acc
        for epoch in range(self.num_epoch):
            # to save time 
            loop = tqdm(self.data.train_dl)
            # gpu = self.hyper_params['gpu']
            self.student.train()
            self.student = self.student.to(gpu)
            if self.teacher is not None:
                self.teacher.eval()
                self.teacher = self.teacher.to(gpu)
            trn = list()
            for images, labels in loop:
                if gpu != 'cpu':
                    images = torch.autograd.Variable(images).to(gpu).float()
                    labels = torch.autograd.Variable(labels).to(gpu)
                else:
                    images = torch.autograd.Variable(images).float()
                    labels = torch.autograd.Variable(labels)

                if quartized:
                    images = images.to('cpu')
                    torch.quantize_per_tensor(images, scale=1e-3, zero_point=0, dtype=torch.quint8)

                y_pred = self.student(images)
                y_pred = y_pred.to('cuda')
                if self.teacher is not None:
                    soft_targets = self.teacher(images)

                # classifier training
                if self.teacher is None:
                    loss = self.loss_function(y_pred, labels)
                # stage training (and assuming sf_teacher and sf_student are given)

                elif self.expt == 'hinton-kd':
                    TEMP = self.hyper_params['temperature']
                    ALPHA = self.hyper_params['alpha']
                    soft_targets = F.softmax(soft_targets/TEMP,dim=1)
                    loss = self.loss_function(F.log_softmax(y_pred/TEMP,dim=1),soft_targets)*(1-ALPHA)*TEMP*TEMP
                    loss += self.loss_function2(F.softmax(y_pred,dim=1),labels)*(ALPHA)

                elif self.loss_function2 is None:
                    if self.expt == 'fsp-kd':
                        loss = 0
                        # 4 intermediate feature maps and taken 2 at a time (thus 3)
                        for k in range(3):
                            loss += self.loss_function(fsp_matrix(self.sf_teacher[k].features, self.sf_teacher[k + 1].features),
                                                fsp_matrix(self.sf_student[k].features, self.sf_student[k + 1].features))
                        loss /= 3
                    else:
                        loss = self.loss_function(self.sf_student[self.hyper_params['stage']].features, self.sf_teacher[self.hyper_params['stage']].features)
                # attention transfer KD
                elif self.expt == 'attention-kd':
                    loss = self.loss_function(y_pred, labels)
                    for k in range(4):
                        loss += self.loss_function2(at(self.sf_student[k].features), at(self.sf_teacher[k].features))
                    loss /= 5
                # 2 loss functions and student and teacher are given -> simultaneous training
                else:
                    loss = self.loss_function(y_pred, labels)
                    for k in range(5):
                        loss += self.loss_function2(self.sf_student[k].features, self.sf_teacher[k].features)
                    # normalizing factor (doesn't affect optimization theoretically)
                    loss /= 6

                trn.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                
                ### added by wenjie for pruning the gradients
                tensor_orig = {} 
                if self.pruning:
                    # zero-out all the gradients corresponding to the pruned connections
                    for name, p in self.student.named_parameters():
                        tensor = p.data.cpu().numpy()
                        tensor_orig[name] = tensor
                        grad_tensor = p.grad.data.cpu().numpy()
                        grad_tensor = np.where(tensor==0, 0, grad_tensor)
                        p.grad.data = torch.from_numpy(grad_tensor).to(gpu)
                                                
                self.optimizer.step()
                
                # for opts like adam, weights are still updated even if gradients are zeros. 
                # so prune weights at each update.
                if self.pruning:
                    for name, p in self.student.named_parameters():
                        tensor = p.data.cpu().numpy()
                        data_tensor = np.where(tensor_orig[name]==0, 0, tensor)
                        p.data = torch.from_numpy(data_tensor).to(gpu)
                
                ### added by yujie for retraining after weight sharing
                if self.bits_weight_sharing:
                    lr = 0.001
                    if self.expt == 'hinton-kd':
                        lr = 0.0001
                    elif self.expt == 'no-teacher':
                        lr = 0.0001
                    for name,i in self.student.named_parameters():
                        i.data[i.kmeans_result == -1] = 0 
            #             print(i.data)
            #             break
                        if i.kmeans_result is None:
                            continue
                        for x in range(2 ** self.bits_weight_sharing):
                            grad = torch.sum(i.grad.detach()[i.kmeans_result == x])
            #                 print(grad.item())
                            i.kmeans_label[x] += -lr * grad.item()
                            i.data[i.kmeans_result == x] = i.kmeans_label[x].item()


                loop.set_description('Epoch {}/{}'.format(epoch + 1, self.num_epoch))         # edited by yujie
                loop.set_postfix(loss=loss.item())

            train_loss = (sum(trn) / len(trn))
            val_loss, val_acc, max_val_acc = self.evaluate(max_val_acc, gpu, quartized)
        return self.student, train_loss, val_loss, val_acc, max_val_acc

    def evaluate(self, max_val_acc=0, gpu='cpu', quartized=False):
        self.student.eval()
        val = list()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, labels) in enumerate(self.data.valid_dl):
                if gpu != 'cpu':
                    images = torch.autograd.Variable(images).to(gpu).float()
                    labels = torch.autograd.Variable(labels).to(gpu)
                else:
                    images = torch.autograd.Variable(images).float()
                    labels = torch.autograd.Variable(labels)

                if quartized:
                    images = images.to('cpu')
                    torch.quantize_per_tensor(images, scale=1e-3, zero_point=0,dtype=torch.quint8)
                y_pred = self.student(images)
                y_pred = y_pred.to('cuda')
                if self.teacher is not None:
                    soft_targets = self.teacher(images)

                # classifier training
                if self.teacher is None:
                    loss = self.loss_function(y_pred, labels)
                    y_pred = F.log_softmax(y_pred, dim = 1)

                    _, pred_ind = torch.max(y_pred, 1)

                    total += labels.size(0)
                    correct += (pred_ind == labels).sum().item()
                
                elif self.expt == 'hinton-kd':
                    ALPHA = self.hyper_params['alpha']

                    y_pred = F.log_softmax(y_pred, dim = 1)
                    _, pred_ind = torch.max(y_pred, 1)

                    total += labels.size(0)
                    correct += (pred_ind == labels).sum().item()

                    loss = self.loss_function2(y_pred,labels)

                # stage training
                elif self.loss_function2 is None:
                    if self.expt == 'fsp-kd':   
                        loss = 0
                        # 4 intermediate feature maps and taken 2 at a time (thus 3)
                        for k in range(3):
                            loss += self.loss_function(fsp_matrix(self.sf_teacher[k].features, self.sf_teacher[k + 1].features),
                                                fsp_matrix(self.sf_student[k].features, self.sf_student[k + 1].features))
                        loss /= 3
                    else: # traditional KD
                        loss = self.loss_function(self.sf_student[self.hyper_params['stage']].features, self.sf_teacher[self.hyper_params['stage']].features)
                        
                # simultaneous training or attention KD
                else:
                    loss = self.loss_function(y_pred, labels)
                    y_pred = F.log_softmax(y_pred, dim = 1)

                    _, pred_ind = torch.max(y_pred, 1)

                    total += labels.size(0)
                    correct += (pred_ind == labels).sum().item()

                val.append(loss.item())

        val_loss = (sum(val) / len(val))
        if total > 0:
            val_acc = correct / total
        else:
            val_acc = None

        # classifier training
        if self.teacher is None:
            if (val_acc * 100) > max_val_acc :
                print(f'higher valid acc obtained: {val_acc * 100}')
                max_val_acc = val_acc * 100
                torch.save(self.student.state_dict(), self.savename)
        # traditional KD, stage training
        elif self.loss_function2 is None:
            if val_loss < max_val_acc :
                print(f'lower valid loss obtained: {val_loss}')
                max_val_acc = val_loss
                torch.save(self.student.state_dict(), self.savename)
        # hinton_kd, simultaneous training or attention kd
        else:
            if (val_acc * 100) > max_val_acc :
                print(f'higher valid acc obtained: {val_acc * 100}')
                max_val_acc = val_acc * 100
                torch.save(self.student.state_dict(), self.savename)

        return val_loss, val_acc, max_val_acc

    def eval_model(self, model, quartized=False, gpu='cpu'):
        model.eval()
        val = list()
        with torch.no_grad():
            correct = 0
            total = 0
            for _, (images, labels) in enumerate(self.data.valid_dl):
                if gpu != 'cpu':
                    images = torch.autograd.Variable(images).to(gpu).float()
                    labels = torch.autograd.Variable(labels).to(gpu)
                else:
                    images = torch.autograd.Variable(images).float()
                    labels = torch.autograd.Variable(labels)
                    
                # edited by yujie
                if quartized:
                    images = images.to('cpu')
                    torch.quantize_per_tensor(images, scale=1e-3, zero_point=0,dtype=torch.quint8)
                y_pred = model(images)
                loss_function = nn.CrossEntropyLoss()
                y_pred = y_pred.to('cuda')
                loss = loss_function(y_pred, labels)    # edited by yujie
                y_pred = F.log_softmax(y_pred, dim = 1)
                _, pred_ind = torch.max(y_pred, 1)
                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()
                val.append(loss.item())
                
            val_loss = (sum(val) / len(val))
            if total > 0:
                val_acc = correct / total
            else:
                val_acc = None
        return val_loss, val_acc
