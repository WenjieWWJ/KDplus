PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python image_classification/experiments/no_teacher.py -d=cifar10 -m=resnet18 -e=100 -s=0
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python image_classification/experiments/traditional_kd.py -d=cifar10 -m=resnet18 -e=100 -s=0
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 nohup python image_classification/experiments/hinton_kd.py -d=cifar10 -m=resnet18 -e=100 -s=0