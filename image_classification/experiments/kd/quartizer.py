from kd.operator import BaseKDOp
import torch.quantization as quantization
import torch 
from kd import kd_util 
from image_classification.models.custom_resnet import BasicBlock

class Quartizer(BaseKDOp):

    def __init__(self, qconfig=quantization.default_qconfig):
        self.qconfig = quantization.default_qconfig

    def apply(self, net:torch.nn.Module, inplace=True) -> torch.nn.Module:
        fused_net = self.fuse_resnet(net)
        print(f"fused net,  size: {kd_util.size_of_model(fused_net)} MB, count of params: {kd_util.count_params(fused_net)}" )
        
        fused_net.qconfig = self.qconfig
        fused_net = quantization.prepare(fused_net, inplace=inplace)
        
        # pytorch qutization does not support GPU for now
        fused_net = fused_net.to('cpu')
        quartized_net = quantization.convert(fused_net, inplace=inplace)
        print(f"quartized net, size: {kd_util.size_of_model(quartized_net)} MB, count of params: {kd_util.count_params(quartized_net)}" )
        return quartized_net

    def fuse_resnet(self, net, inplace=True):
        quantization.fuse_modules(net, [['conv1', 'bn1', 'relu']], inplace=inplace)
        for m in net.modules():
            if type(m) == torch.nn.Sequential and type(m[0]) == BasicBlock:
                quantization.fuse_modules(m[0], [['conv1', 'bn1', 'relu'],['conv2','bn2']], inplace=inplace)
        return net