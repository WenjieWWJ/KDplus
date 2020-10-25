from comet_ml import Experiment
from tqdm import tqdm

from fastai.vision import *
### added by Wenjie (Oct 19)
import warnings
warnings.filterwarnings("ignore")
###
from image_classification.utils.utils import *

class KDTrainer:

    def __init__(self, student, teacher, data, sf_teacher, sf_student, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, expt=None):
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
        self.expt = expt

    def train(self):
        # to keep consistent with the original train  method
        
        max_val_acc = self.best_val_acc
        for epoch in range(self.num_epoch):
            # to save time 
            loop = tqdm(self.data.train_dl)
            gpu = self.hyper_params['gpu']
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

                y_pred = self.student(images)
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
                    loss = loss_function(F.log_softmax(y_pred/TEMP,dim=1),soft_targets)*(1-ALPHA)*TEMP*TEMP
                    loss += loss_function2(F.softmax(y_pred,dim=1),labels)*(ALPHA)


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
                self.optimizer.step()

                loop.set_description('Epoch {}/{}'.format(epoch + 1, self.hyper_params['num_epochs']))
                loop.set_postfix(loss=loss.item())

            train_loss = (sum(trn) / len(trn))
            val_loss, val_acc, max_val_acc = self.evaluate(max_val_acc, gpu)
        return self.student, train_loss, val_loss, val_acc, max_val_acc

    def evaluate(self, max_val_acc=0, gpu='cpu'):
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

                y_pred = self.student(images)
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
                    if expt == 'fsp-kd':
                        loss = 0
                        # 4 intermediate feature maps and taken 2 at a time (thus 3)
                        for k in range(3):
                            loss += self.loss_function(fsp_matrix(self.sf_teacher[k].features, self.sf_teacher[k + 1].features),
                                                fsp_matrix(self.sf_student[k].features, self.sf_student[k + 1].features))
                        loss /= 3
                    else:
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
        # stage training
        elif self.loss_function2 is None:
            if val_loss < max_val_acc :
                print(f'lower valid loss obtained: {val_loss}')
                max_val_acc = val_loss
                torch.save(self.student.state_dict(), self.savename)
        # simultaneous training or attention kd
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
                
            if quartized:
                images = images.to('cpu')
                torch.quantize_per_tensor(images, scale=1e-3, zero_point=128,dtype=torch.quint8)

            y_pred = model(images)
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
        return val_loss, val_acc