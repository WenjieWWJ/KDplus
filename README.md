# [Pruning and Quantization Enhanced knowledge Distillation](/CS6285_Project_Quantization_Enhanced_KD.pdf)

CS6285 Project By Wenjie Wang, Fengbin Zhu, Yujie Zhang, and Yichen Zhou.

Deep compression is important for modern deep learning models in various industrial applications. There are several different types of deep compression techniques that aim to reduce the size of the network and accelerate the model inference, such as Knowledge Distillation (KD), Pruning, and Quantization. However, existing works seldom study the combination of multiple different techniques. In this work, we aim to explore and verify the effectiveness of combining different KD methods with various compression techniques including pruning, weight sharing and quantization. We conducted comprehensive experiments for several compression pipelines with two or three compression steps on CIFAR10 with ResNets. We demonstrate that pruning and quantization enhanced KD can further compress the student model while maintaining the performance. Besides, KD methods perform differently when incorporating various compression techniques. The insights shed light on how to effectively incorporate various deep compression techniques when training deep learning models.

![avatar](/pipeline.pdf)



## Requirements
- Install the dependencies using `conda` with the `requirements.yml` file
    ```
    conda env create -f environment.yml
    ```
- Setup the `stagewise-knowledge-distillation` package itself
    ```
    pip install -e .
    ```
- Apart from the above mentioned dependencies, it is recommended to have an Nvidia GPU (CUDA compatible) with at least 8 GB of video memory (most of the experiments will work with 6 GB also). However, the code works with CPU only machines as well.

## Image Classification
### Introduction
In this work, [ResNet](https://arxiv.org/abs/1512.03385) architectures are used. Particularly, we used ResNet10, 14, 18, 20 and 26 as student networks and ResNet34 as the teacher network. The datasets used are [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [Imagenette](https://github.com/fastai/imagenette) and [Imagewoof](https://github.com/fastai/imagenette). Note that Imagenette and Imagewoof are subsets of [ImageNet](http://www.image-net.org/).

### Preparation
- Before any experiments, you need to download the data and saved weights of teacher model to appropriate locations. 
- The following script
    - downloads the datasets
    - saves 10%, 20%, 30% and 40% splits of each dataset separately
    - downloads teacher model weights for all 3 datasets

    ```
    # assuming you are in the root folder of the repository
    cd image_classification/scripts
    bash setup.sh
    ```
    
### Experiments
For detailed information on the various experiments, refer to the paper. In all the image classification experiments, the following common training arguments are listed with the possible values they can take:
- dataset (`-d`) : imagenette, imagewoof, cifar10
- model (`-m`) : resnet10, resnet14, resnet18, resnet20, resnet26, resnet34
- number of epochs (`-e`) : Integer is required
- percentage of dataset (`-p`) : 10, 20, 30, 40 (don't use this argument at all for full dataset experiments)
- random seed (`-s`) : Give any random seed (for reproducibility purposes)
- gpu (`-g`) : Don't use unless training on CPU (in which case, use `-g 'cpu'` as the argument). In case of multi-GPU systems, run `CUDA_VISIBLE_DEVICES=id` in the terminal before the experiment, where `id` is the ID of your GPU according to `nvidia-smi` output.
- Comet ML API key (`-a`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your API key as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.
- Comet ML workspace (`-w`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your workspace name as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.

We tested the No-teacher, FitNets, and HintonKD with pruning, weight sharing, and quantization, respectively. 
In addition, we also tried futher compression: 
1. KD+QAT+pruning;
2. KD+pruning+weight sharing.

In the following subsections, example commands for training are given for one experiment each.
#### No Teacher
Full Imagenette dataset, ResNet10
```
python3 no_teacher.py -d imagenette -m resnet10 -e 100 -s 0
```

#### Traditional KD ([FitNets](https://arxiv.org/abs/1412.6550))
20% Imagewoof dataset, ResNet18
```
python3 traditional_kd.py -d imagewoof -m resnet18 -p 20 -e 100 -s 0
```

#### [Hinton KD](https://arxiv.org/abs/1503.02531)
Full CIFAR10 dataset, ResNet14
```
python3 hinton_kd.py -d cifar10 -m resnet14 -e 100 -s 0
```

#### FitNets with Prunning
```
python3 traditional_kd_pruning.py -d cifar10 -m resnet18 -p 20 -e 100 -s 0
```

More testing files can be found in ./image_classification/experiments/.


## Acknowledgment

Thanks to the KD implementation in [stageKD](https://github.com/IvLabs/stagewise-knowledge-distillation), built by [Akshay Kulkarni](https://akshayk07.weebly.com/), [Navid Panchi](https://navidpanchi.wixsite.com/home) and [Sharath Chandra Raparthy](https://sharathraparthy.github.io/).
