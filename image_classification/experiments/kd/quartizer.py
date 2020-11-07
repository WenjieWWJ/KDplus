from kd.operator import BaseKDOp
import torch 
import numpy as np
from kd import kd_util 
from image_classification.models.custom_resnet import BasicBlock
from sklearn.cluster import KMeans      # edited by yujie
# import torch.nn.utils.prune as prune

class Quartizer(BaseKDOp):

    def __init__(self, trainer=None, qat=False):
        torch.backends.quantized.engine = 'fbgemm'
        self.trainer = trainer
        self.qat = qat
        # self.hyper_params = hyper_params
        if qat:
            self.qconfig = torch.quantization.default_qat_qconfig
        else:
            self.qconfig = torch.quantization.default_qconfig

    def apply(self, net:torch.nn.Module, inplace=True) -> torch.nn.Module:
        fused_net = self.fuse_resnet(net)
        fused_net.qconfig = torch.quantization.default_qconfig
        if self.qat:
            fused_net = torch.quantization.prepare_qat(fused_net, inplace=inplace)
            return fused_net
        else:
            fused_net = fused_net.to('cpu')
            fused_net = torch.quantization.prepare(fused_net, inplace=inplace)
            fused_net.eval()
            val_loss, val_acc = self.trainer.eval_model(fused_net, quartized=True, gpu="cpu")
            print(f"fused net,  size: {kd_util.size_of_model(fused_net)} MB, count of params: {kd_util.count_params(fused_net)}, val_loss: {val_loss}, val_acc: {val_acc}"  )
            # pytorch qutization does not support GPU for now
            quartized_net = torch.quantization.convert(fused_net, inplace=inplace)
            return quartized_net

    def fuse_resnet(self, net):
        net.fuse_model()
        return net

 
# # edited by yujie
# def apply_weight_sharing(model, bits=5):    

#     for name, parameter in model.named_parameters():
#         if 'conv' in name or 'fc.weight' in name or 'fc2.weight' in name:
#             print('name:', name)
#             print('parameters:', parameter.size())
#             dev = parameter.device
#             weight = parameter.data.cpu().numpy()
#             shape = weight.shape
#             print('shape:', shape)
#             weight_scope = weight.reshape(-1)
#             min_ = min(weight_scope)
#             max_ = max(weight_scope)
#             space = np.linspace(min_, max_, num=2**bits)
#             kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
#             kmeans.fit(weight.reshape(-1,1))
#             new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
#             weight.data = new_weight
#             parameter.data = torch.from_numpy(weight).to(dev)

# edited by yujie
def apply_weight_sharing(model, bits=5):  

    kmean_list = []

    for name, parameter in model.named_parameters():
        data = parameter.data.clone().view(-1).cpu().detach().numpy().reshape(-1)
        data = data[data != 0]
        if data.size < 2**bits: 
            kmean_list.append(None)
            continue
        init = [x*(np.max(data)+np.min(data))/(2 ** bits) + np.min(data) for x in range(2 ** bits)]
        kmn = KMeans(2 ** bits, init=np.array(init).reshape(2 ** bits, 1))
        kmn.fit(data.reshape((-1,1)))        
        kmean_list.append(kmn)
        print(name,parameter.shape)


    for i,(name,f) in enumerate(model.named_parameters()):
        data = f.data.clone().view(-1).cpu().detach().numpy().reshape(-1)
        data_nozero = data[data != 0].reshape((-1,1))
        if data_nozero.size == 0 or data.size < 2 ** bits or kmean_list[i] is None:
            f.kmeans_result = None
            f.kmeans_label = None
            continue
    #     print(name)
    #     print(data.size)
    
        result = data.copy()
        result[result == 0] = -1
        
#         print(data_nozero)
#         print(kmean_list[i])
        label = kmean_list[i].predict(data_nozero).reshape(-1)
#         print(data_nozero)
#         print(label)
        new_data = np.array([kmean_list[i].cluster_centers_[x] for x in label])
        data[data != 0] = new_data.reshape(-1)
#         print(data,new_data)
        f.data = torch.from_numpy(data).view(f.data.shape).cuda()
        result[result != -1] = label
        f.kmeans_result = torch.from_numpy(result).view(f.data.shape).cuda()
        f.kmeans_label = torch.from_numpy(kmean_list[i].cluster_centers_).cuda()   


# edited by wenjie
def apply_pruning(model, prune_ratio):
    
    # for now, we only prune the weights of fc, conv, bn, and downsample layers. (no pruning for bias)
    parameters_to_prune = tuple([(module, 'weight') for name, module in model.named_modules() if kd_util.check_name_prune(name)])

    kd_util.print_model_parameters(model)
#     print(parameters_to_prune)
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=prune_ratio,
    )
    # assign the pruned weight with zero permanently
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    
    
    
    
    
    
