from models.LRRNet import *

def get_model(name, net=None):
    if name == 'LRRNet':
        net = UNet()
    else:
        raise NotImplementedError

    return net

