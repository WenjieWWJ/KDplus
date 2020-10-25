import torch
import os

def size_of_model(net: torch.nn.Module):
    torch.save(net.state_dict(), "temp.p")
    size =  os.path.getsize("temp.p")/1e6 
    print('Size (MB):', size)
    os.remove('temp.p')
    return size

def count_params(net: torch.nn.Module):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)