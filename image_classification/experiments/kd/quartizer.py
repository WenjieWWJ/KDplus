from kd.operator import BaseKDOp
import torch.quantization as quantization
import torch 
from kd import kd_util 
from image_classification.models.custom_resnet import BasicBlock
from sklearn.cluster import KMeans      # edited by yujie

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
 
# edited by yujie
def apply_weight_sharing(model, bits=5):    

    for name, parameter in model.named_parameters():
        if 'conv' in name or 'fc.weight' in name or 'fc2.weight' in name:
            print('name:', name)
            print('parameters:', parameter.size())
            dev = parameter.device
            weight = parameter.data.cpu().numpy()
            shape = weight.shape
            print('shape:', shape)
            weight_scope = weight.reshape(-1)
            min_ = min(weight_scope)
            max_ = max(weight_scope)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(weight.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            weight.data = new_weight
            parameter.data = torch.from_numpy(weight).to(dev)
   
