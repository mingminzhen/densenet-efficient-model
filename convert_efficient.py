import torch
import torch.nn as nn
import DenseNet_Efficient
def densenet_264_k32_efficient_model():
    model_dict = torch.load('densenet_264_k32.pth')
    v1 =[]
    k1 =[]
    for k,v in model_dict.items():
        v1.append(v)
        k1.append(k)
    # generate efficient model
    growth_rate = 32
    block_config=(6,12,64,48)
    model = DenseNet_Efficient.DenseNetEfficient(num_init_features=64,
                                                 growth_rate = growth_rate,
                                                block_config = block_config,
                                                 num_classes = 1000,
                                                       cifar = False)
    k2 =[]
    v2 =[]
    for key, value in model.state_dict().items():
        k2.append(key)
        v2.append(value)
        pretrained_dict = {}
    idx = 0
    while(idx < len(k2)):
        if k2[idx].endswith('bn.conv_weight'):
            pretrained_dict[k2[idx]] = v1[idx+2]
            pretrained_dict[k2[idx+1]] = v1[idx]
            pretrained_dict[k2[idx+2]] = v1[idx+1]
            print('{}: {}->{}     {}-->{}'.format(idx,k1[idx+2],k2[idx],v1[idx+2].shape,v2[idx].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+1,k1[idx],k2[idx+1],v1[idx].shape,v2[idx+1].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+2, k1[idx+1],k2[idx+2],v1[idx+1].shape,v2[idx+2].shape))
            idx += 3
        else:
            pretrained_dict[k2[idx]] = v1[idx]
            print('{}: {}->{}     {}-->{}'.format(idx, k1[idx],k2[idx],v1[idx].shape,v2[idx].shape))
            idx += 1
    model.load_state_dict(pretrained_dict)
    torch.save(model.state_dict(),'densenet_264_k32_eff.pth')
    return 

def densenet_232_k48_efficient_model():
    model_dict = torch.load('densenet_232_k48.pth')
    v1 =[]
    k1 =[]
    for k,v in model_dict.items():
        v1.append(v)
        k1.append(k)
    # generate efficient model
    growth_rate = 48
    block_config=(6,12,48,48)
    model = DenseNet_Efficient.DenseNetEfficient(num_init_features=96,
                                                 growth_rate = growth_rate,
                                                block_config = block_config,
                                                 num_classes = 1000,
                                                       cifar = False)
    k2 =[]
    v2 =[]
    for key, value in model.state_dict().items():
        k2.append(key)
        v2.append(value)
        pretrained_dict = {}
    idx = 0
    while(idx < len(k2)):
        if k2[idx].endswith('bn.conv_weight'):
            pretrained_dict[k2[idx]] = v1[idx+2]
            pretrained_dict[k2[idx+1]] = v1[idx]
            pretrained_dict[k2[idx+2]] = v1[idx+1]
            print('{}: {}->{}     {}-->{}'.format(idx,k1[idx+2],k2[idx],v1[idx+2].shape,v2[idx].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+1,k1[idx],k2[idx+1],v1[idx].shape,v2[idx+1].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+2, k1[idx+1],k2[idx+2],v1[idx+1].shape,v2[idx+2].shape))
            idx += 3
        else:
            pretrained_dict[k2[idx]] = v1[idx]
            print('{}: {}->{}     {}-->{}'.format(idx, k1[idx],k2[idx],v1[idx].shape,v2[idx].shape))
            idx += 1
    model.load_state_dict(pretrained_dict)
    torch.save(model.state_dict(),'densenet_232_k48_eff.pth')
    return 

def densenet_cosine_264_k32_efficient_model():
    model_dict = torch.load('densenet_cosine_264_k32.pth')
    v1 =[]
    k1 =[]
    for k,v in model_dict.items():
        v1.append(v)
        k1.append(k)
    # generate efficient model
    growth_rate = 32
    block_config=(6,12,64,48)
    model = DenseNet_Efficient.DenseNetEfficient(num_init_features=64,
                                                 growth_rate = growth_rate,
                                                block_config = block_config,
                                                 num_classes = 1000,
                                                       cifar = False)
    k2 =[]
    v2 =[]
    for key, value in model.state_dict().items():
        k2.append(key)
        v2.append(value)
        pretrained_dict = {}
    idx = 0
    while(idx < len(k2)):
        if k2[idx].endswith('bn.conv_weight'):
            pretrained_dict[k2[idx]] = v1[idx+2]
            pretrained_dict[k2[idx+1]] = v1[idx]
            pretrained_dict[k2[idx+2]] = v1[idx+1]
            print('{}: {}->{}     {}-->{}'.format(idx,k1[idx+2],k2[idx],v1[idx+2].shape,v2[idx].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+1,k1[idx],k2[idx+1],v1[idx].shape,v2[idx+1].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+2, k1[idx+1],k2[idx+2],v1[idx+1].shape,v2[idx+2].shape))
            idx += 3
        else:
            pretrained_dict[k2[idx]] = v1[idx]
            print('{}: {}->{}     {}-->{}'.format(idx, k1[idx],k2[idx],v1[idx].shape,v2[idx].shape))
            idx += 1
    model.load_state_dict(pretrained_dict)
    torch.save(model.state_dict(),'densenet_cosine_264_k32_eff.pth')
    return 

def densenet_cosine_264_k48_efficient_model():
    model_dict = torch.load('densenet_cosine_264_k48.pth')
    v1 =[]
    k1 =[]
    for k,v in model_dict.items():
        v1.append(v)
        k1.append(k)
    # generate efficient model
    growth_rate = 48
    block_config=(6,12,64,48)
    model = DenseNet_Efficient.DenseNetEfficient(num_init_features=96,
                                                 growth_rate = growth_rate,
                                                block_config = block_config,
                                                 num_classes = 1000,
                                                       cifar = False)
    k2 =[]
    v2 =[]
    for key, value in model.state_dict().items():
        k2.append(key)
        v2.append(value)
        pretrained_dict = {}
    idx = 0
    while(idx < len(k2)):
        if k2[idx].endswith('bn.conv_weight'):
            pretrained_dict[k2[idx]] = v1[idx+2]
            pretrained_dict[k2[idx+1]] = v1[idx]
            pretrained_dict[k2[idx+2]] = v1[idx+1]
            print('{}: {}->{}     {}-->{}'.format(idx,k1[idx+2],k2[idx],v1[idx+2].shape,v2[idx].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+1,k1[idx],k2[idx+1],v1[idx].shape,v2[idx+1].shape))
            print('{}: {}->{}     {}-->{}'.format(idx+2, k1[idx+1],k2[idx+2],v1[idx+1].shape,v2[idx+2].shape))
            idx += 3
        else:
            pretrained_dict[k2[idx]] = v1[idx]
            print('{}: {}->{}     {}-->{}'.format(idx, k1[idx],k2[idx],v1[idx].shape,v2[idx].shape))
            idx += 1
    model.load_state_dict(pretrained_dict)
    torch.save(model.state_dict(),'densenet_cosine_264_k48_eff.pth')
    return 
def main():
    ####### convert  densenet_cosine_264_k48 to efficient model
    # densenet_cosine_264_k48_efficient_model()
    ####### convert  densenet_cosine_264_k32 to efficient model
    # densenet_cosine_264_k32_efficient_model()
    ####### convert  densenet_232_k48 to efficient model
    # densenet_232_k48_efficient_model()
    ####### convert  densenet_264_k32 to efficient model
    densenet_264_k32_efficient_model()
if __name__ == '__main__':
    main()

