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
from image_classification.models.custom_resnet import BasicBlock
from kd_trainer import KDTrainer
from kd.quartizer import Quartizer

args = get_args(description='No Teacher', mode='train')
expt = 'no-teacher'

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

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

net = get_model(hyper_params['model'], hyper_params['dataset'])
net = net.to(args.gpu)

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

optimizer = torch.optim.Adam(net.parameters(), lr=hyper_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()
savename = get_savename(hyper_params, experiment=expt)
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
student, train_loss, val_loss, val_acc, best_val_acc = trainer.train()
if args.api_key:
    experiment.log_metric("train_loss", train_loss)
    experiment.log_metric("val_loss", val_loss)
    experiment.log_metric("val_acc", val_acc * 100)


# ======= Below are customized KD & DC code ==========
net.eval()
val_loss, val_acc = trainer.eval_model(model=net, quartized=False)
print(f"original net, val_loss: {val_loss}, val_acc: {val_acc} ")
qtz = Quartizer()
quartized_net = qtz.apply(net)
val_loss, val_acc = trainer.eval_model(model=quartized_net, quartized=True)
print(f"quartized net, val_loss: {val_loss}, val_acc: {val_acc}")
