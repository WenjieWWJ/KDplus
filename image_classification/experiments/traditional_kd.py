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
    "gpu": args.gpu
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

teacher = learn.model

sf_student, sf_teacher = get_features(net, teacher, experiment=expt)

for stage in range(2):
    if stage != 0:
        # load previous stage best weights
        filename = get_savename(hyper_params, experiment=expt)
        net.load_state_dict(torch.load(filename))
    
    hyper_params['stage'] = stage
    
    print('-------------------------------')
    print('stage :', hyper_params['stage'])
    print('-------------------------------')
    
    net = freeze_student(net, hyper_params, experiment=expt)
    
    if args.api_key:
        project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
        experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
        experiment.log_parameters(hyper_params)
    
    savename = get_savename(hyper_params, experiment=expt)
    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
    
    if hyper_params['stage'] != 1:
        loss_function = nn.MSELoss()
        best_val_loss = 100   
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

# ======= Below are customized KD & DC code ==========
net.eval()
val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
print(f"original net_0, val_loss: {val_loss}, val_acc: {val_acc} ")

apply_weight_sharing(net, bits=5)
val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
print(f"net_1 after weight sharing, val_loss: {val_loss}, val_acc: {val_acc}")

loss_function = nn.CrossEntropyLoss()
best_val_acc = 0  
trainer_after_weightSharing = KDTrainer(net,
                                        teacher=None,
                                        data=data,
                                        sf_teacher=None,
                                        sf_student=None,
                                        loss_function=loss_function,
                                        loss_function2=None,
                                        optimizer=optimizer,
                                        hyper_params=hyper_params,
                                        epoch=30,
                                        savename=savename,
                                        best_val_acc=best_val_acc)
net, train_loss, val_loss, val_acc, best_val_acc = trainer_after_weightSharing.train()

net.eval()
val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
print(f"net_2 after retraining net_1, val_loss: {val_loss}, val_acc: {val_acc} ")
