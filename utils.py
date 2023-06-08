import time

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.cifar100 import CIFAR100
from datasets.cub200 import CUB200
from datasets.flowers102 import Flowers102
from datasets.fgvcaircraft import FGVCAircraft

from models.quantizer import uniform, minteq, posteq



dataset_path = 'datasets/files/'
dataset_args = {'CIFAR100':     [CIFAR100,
                                 32,
                                 [0.5070757865905762, 0.48655030131340027, 0.4409191310405731],
                                 [0.2673342823982239, 0.2564384639263153, 0.2761504650115967],
                                 100],
                'CUB200':       [CUB200,
                                 448,
                                 [0.4856198728084564, 0.49942857027053833, 0.43238958716392517],
                                 [0.22942277789115906, 0.22484524548053741, 0.2633439004421234],
                                 200],
                'Flowers102':   [Flowers102,
                                 448,
                                 [0.43443986773490906, 0.38297420740127563, 0.2954137921333313],
                                 [0.2925998270511627, 0.24457821249961853, 0.2723919749259949],
                                 102],
                'FGVCAircraft': [FGVCAircraft,
                                 448,
                                 [0.47963711619377136, 0.5107827186584473, 0.5341059565544128],
                                 [0.221391960978508, 0.21462024748325348, 0.2460494488477707],
                                 100]}
weight_quant = {'Uniform': uniform,
                'MinTeQ': minteq,
                'PosTeQ': posteq}
ratios_dict = {3: {2: [[0.500, 0.250, 0.250],
                       [0.250, 0.250, 0.500]],
                   4: [[0.625, 0.250, 0.125],
                       [0.500, 0.250, 0.250],
                       [0.250, 0.250, 0.500],
                       [0.125, 0.250, 0.625]]},
               5: {2: [[0.250, 0.375, 0.125, 0.125, 0.125],
                       [0.125, 0.125, 0.125, 0.375, 0.250]],
                   4: [[0.375, 0.250, 0.125, 0.125, 0.125],
                       [0.250, 0.375, 0.125, 0.125, 0.125],
                       [0.125, 0.125, 0.125, 0.375, 0.250],
                       [0.125, 0.125, 0.125, 0.250, 0.375]]}}



def get_transform(split, size, mean, std):
    if split == 'train':
        return transforms.Compose([transforms.Resize(size),
                                   transforms.RandomCrop(size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    elif split == 'test':
        return transforms.Compose([transforms.Resize(size),
                                   transforms.CenterCrop(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])



def get_dataset(args, split):
    dataset_func, size, mean, std = dataset_args[args.dataset][:4]
    device = 'cuda' if args.gpu else 'cpu'
    transform = get_transform(split, size, mean, std)
    dataset = dataset_func(dataset_path, split, transform=transform, download=True)
    dataset = [(X.to(device), torch.tensor(y).to(device)) for X, y in dataset]
    return dataset



def get_dataloader(args, splits):
    print('Preparing dataloader...')

    start = time.time()


    dataloaders = []
    for split in splits:
        dataset = get_dataset(args, split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloaders.append(dataloader)


    finish = time.time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(finish - start), end='\n\n')
    
    return dataloaders



def get_model(args):
    print('Preparing model...')

    start = time.time()


    if args.model == 'VGG19':
        from models.vgg import vgg19
        model_func = vgg19
    elif args.model == 'B-CNN':
        from models.b_cnn import b_cnn
        model_func = b_cnn


    in_size, num_class = dataset_args[args.dataset][1], dataset_args[args.dataset][4]
    model = model_func(args, in_size, num_class)


    finish = time.time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(finish - start), end='\n\n')

    return model



def load_one_epoch(split, loader, model, loss_func, optimizer):
    start = time.time()


    if split == 'train':
        model.train()
    else:
        model.eval()

    avg_loss = 0
    accuracy = 0
    num_data = 0
    for X, y in loader:
        if split == 'train':
            optimizer.zero_grad()

        prediction = model(X)
        loss = loss_func(prediction, y)

        if split == 'train':
            loss.backward()
            optimizer.step()

        avg_loss += loss.item()
        accuracy += (prediction.argmax(dim=1) == y).sum().item()
        num_data += X.shape[0]
    avg_loss /= num_data
    accuracy = (100 * accuracy) / num_data


    finish = time.time()

    return avg_loss, accuracy, finish - start



def quantize_model(model, quantizer, precision, k, m, flipped):
    if quantizer == 'Base':
        return

    print('Quantizing model...')

    start = time.time()


    if quantizer == 'Uniform':
        quant_info = (precision,)
    else:
        if k == 3:
            precisions = [precision-2, precision, precision+2]
        elif k == 5:
            precisions = [precision-2, precision-1, precision, precision+1, precision+2]
        ratios = ratios_dict[k][m]
        if flipped:
            ratios.reverse()

    class quant(nn.Module):
        def __init__(self, precision): super().__init__(); self.precision = precision
        def forward(self, x): return uniform(x, self.precision)

    model.quant = quant(precision)
    model.dequant = quant(precision)

    conv_params, fc_params = [], []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            conv_params.append(param)
        elif 'classifier' in name:
            fc_params.append(param)
    
    l = len(conv_params)
    for i, conv_param in enumerate(conv_params):
        if quantizer != 'Uniform':
            idx = (i * m) // l
            quant_info = (precisions, ratios[idx])
        conv_param.data = weight_quant[quantizer](conv_param.data, *quant_info)
    for fc_param in fc_params:
        fc_param.data = weight_quant['Uniform'](fc_param.data, precision)
    

    finish = time.time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(finish - start), end='\n\n')
