import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from comet_ml import Experiment
from fastai.vision import *
import torch
import argparse
import os
from image_classification.arguments import get_args
from image_classification.datasets.dataset import get_dataset
from image_classification.utils.utils import *
from image_classification.models.custom_resnet import *
from kd_trainer import KDTrainer
from kd.quartizer import *
import random


args = get_args(description='Traditional KD', mode='train')
expt = 'traditional-kd'
print(args)

random.seed(args.seed)  # random and transforms
torch.backends.cudnn.deterministic=True  # cudnn
torch.manual_seed(args.seed)

if args.gpu != 'cpu':
    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "stage": 0,
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu,
    "p_prune": args.prune_percentage,
    "bits": args.bits_weight_sharing   
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

savename = get_savename(hyper_params, experiment=expt)

def train():
    learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
    learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

    teacher = learn.model

    sf_student, sf_teacher = get_features(net, teacher, experiment=expt)

    for stage in range(2):
        if stage != 0:
            net.load_state_dict(torch.load(savename))
        
        hyper_params['stage'] = stage
        
        print('-------------------------------')
        print('stage :', hyper_params['stage'])
        print('-------------------------------')
        
        net = freeze_student(net, hyper_params, experiment=expt)
        
        if args.api_key:
            project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
            experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
            experiment.log_parameters(hyper_params)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
        
        if hyper_params['stage'] != 1:
            loss_function = nn.MSELoss()
            best_val_acc = 100
            # refactor it to a trainer 
            trainer = KDTrainer(net,
                                teacher,
                                data,
                                sf_teacher,
                                sf_student,
                                loss_function,
                                loss_function2=None,
                                optimizer=optimizer,
                                hyper_params=hyper_params,
                                epoch=hyper_params['num_epochs'],
                                savename=savename,
                                best_val_acc=best_val_acc)
            net, train_loss, val_loss, val_acc, best_val_acc = trainer.train()

            if args.api_key:
                experiment.log_metric("train_loss", train_loss)
                experiment.log_metric("val_loss", val_loss)
                experiment.log_metric("val_acc", val_acc * 100)
        else:
            loss_function = nn.CrossEntropyLoss()
            best_val_acc = 0  
            # refactor it to a trainer 
            trainer = KDTrainer(net,
                                teacher=None,
                                data=data,
                                sf_teacher=None,
                                sf_student=None,
                                loss_function=loss_function,
                                loss_function2=None,
                                optimizer=optimizer,
                                hyper_params=hyper_params,
                                epoch=hyper_params['num_epochs'],
                                savename=savename,
                                best_val_acc=best_val_acc)
            net, train_loss, val_loss, val_acc, best_val_acc = trainer.train()

            if args.api_key:
                experiment.log_metric("train_loss", train_loss)
                experiment.log_metric("val_loss", val_loss)
                experiment.log_metric("val_acc", val_acc * 100)


# ======= Below are customized KD & DC code ==========
def eval_quantization():
    loss_function = nn.CrossEntropyLoss()
    # reload the best model
    net = get_model(hyper_params['model'], hyper_params['dataset'])
    net = net.to(args.gpu)
    net.load_state_dict(torch.load(savename))
    net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
    trainer = KDTrainer(net,
                        teacher=None,
                        data=data,
                        sf_teacher=None,
                        sf_student=None,
                        loss_function=loss_function,
                        loss_function2=None,
                        optimizer=optimizer,
                        hyper_params=hyper_params,
                        epoch=hyper_params['num_epochs'],
                        savename=savename,
                        best_val_acc=0)
    val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
    print(f"original net_0 ----, size:: {kd_util.size_of_model(net)} MB val_loss: {val_loss}, val_acc: {val_acc} ")

    # # # # # ! static quartize Quartizer by fengbin
    qtz = Quartizer(trainer=trainer, qat=False)
    quartized_net = qtz.apply(net, inplace=False)
    val_loss, val_acc = trainer.eval_model(model=quartized_net, quartized=True)
    print(f"static quartized - eval ----, size: {kd_util.size_of_model(quartized_net)} MB, count of params: {kd_util.count_params(quartized_net)}, val_loss: {val_loss}, val_acc: {val_acc}" )

def retrain_quantization():
    loss_function = nn.CrossEntropyLoss()
    # ! QAT training the model 
    qat_net = get_model(hyper_params['model'], hyper_params['dataset'])
    qat_net = qat_net.to(args.gpu)
    qat_net.load_state_dict(torch.load(savename))
    qat_net.train()

    qat_qtz = Quartizer(qat=True)
    qat_fused_net = qat_qtz.apply(qat_net)
    # qat_fused_net = qat_fused_net.to('cpu')
    # val_loss, val_acc = trainer.eval_model(model=qat_fused_net, quartized=True)
    # print(f"qat_quartized_net , size: {kd_util.size_of_model(qat_fused_net)} MB, count of params: {kd_util.count_params(qat_fused_net)}, val_loss: {val_loss}, val_acc: {val_acc}" )

    for param in qat_fused_net.parameters():
        param.requires_grad = True
    # kd_util.print_model_parameters(qat_fused_net)

    optimizer = torch.optim.Adam(qat_fused_net.parameters(), lr=hyper_params["learning_rate"])
    qat_savename = get_savename(hyper_params, experiment=expt, quantized=True)

    qat_trainer = KDTrainer(qat_fused_net,
                        teacher=None,
                        data=data,
                        sf_teacher=None,
                        sf_student=None,
                        loss_function=loss_function,
                        loss_function2=None,
                        optimizer=optimizer,
                        hyper_params=hyper_params,
                        epoch=hyper_params['num_epochs'],
                        savename=qat_savename,
                        best_val_acc=0)

    net, train_loss, val_loss, val_acc, best_val_acc = qat_trainer.train(quartized=False, gpu=args.gpu)
    print(f"quartized retraining float32: train_loss,{train_loss}, val_loss: {val_loss}, val_acc: {val_acc}, best_val_acc:{best_val_acc} ")

    qat_fused_net = qat_fused_net.to('cpu')
    qat_fused_net.load_state_dict(torch.load(qat_savename))
    qat_fused_net.eval()
    qat_quartized_net_int8 = torch.quantization.convert(qat_fused_net, inplace=False)

    val_loss, val_acc = qat_trainer.eval_model(model=qat_quartized_net_int8, quartized=True)
    print(f"quartized retraining int8 ----:, size: {kd_util.size_of_model(qat_quartized_net_int8)} MB, count of params: {kd_util.count_params(qat_quartized_net_int8)}, val_loss: {val_loss}, val_acc: {val_acc}" )

print(f"===origin training start ====")
train()
print(f"===evaluate quantization start ====")
eval_quantization()
print(f"===retraining quantization start ====")
retrain_quantization()


# apply_weight_sharing(net, bits=5)
# val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
# print(f"net_1 after weight sharing, val_loss: {val_loss}, val_acc: {val_acc}")

# loss_function = nn.CrossEntropyLoss()
# best_val_acc = 0  
# trainer_after_weightSharing = KDTrainer(net,
#                                         teacher=None,
#                                         data=data,
#                                         sf_teacher=None,
#                                         sf_student=None,
#                                         loss_function=loss_function,
#                                         loss_function2=None,
#                                         optimizer=optimizer,
#                                         hyper_params=hyper_params,
#                                         epoch=hyper_params['num_epochs'],
#                                         savename=savename,
#                                         best_val_acc=best_val_acc)
# net, train_loss, val_loss, val_acc, best_val_acc = trainer_after_weightSharing.train()


# net.load_state_dict(torch.load(savename))

# net.eval()
# val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
# print(f"net_2 after retraining net_1, val_loss: {val_loss}, val_acc: {val_acc} ")
