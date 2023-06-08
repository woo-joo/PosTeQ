from torch import nn
from torch import optim

from utils import load_one_epoch



class VGG(nn.Module):
    def __init__(self, features, in_size, num_class, batch_normalize):
        super().__init__()

        self.batch_normalize = batch_normalize

        self.quant = lambda x: x
        self.feature_extractor = self.make_layers(features)
        self.classifier = nn.Sequential(
                                        nn.Linear(512 * (in_size // 32) ** 2, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_class)
                                        )
        self.dequant = lambda x: x


    def make_layers(self, features):
        layers = []

        in_channels = 3
        for out_channels in features:
            if out_channels == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if self.batch_normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels
        
        return nn.Sequential(*layers)


    def forward(self, x):
        output = self.quant(x)
        output = self.feature_extractor(output)
        output = output.view(output.shape[0], -1)
        output = self.classifier(output)
        output = self.dequant(output)
        
        return output



class VGGManager:
    def __init__(self, features, in_size, num_class, batch_normalize, gpu, learning_rate):
        super().__init__()

        self.model = VGG(features, in_size, num_class, batch_normalize)
        if gpu:
            self.model = self.model.cuda()

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        self.milestones = [60, 120, 160]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.2)


    def train(self, train_loader):
        train_loss, train_accuracy, train_time = load_one_epoch('train', train_loader, self.model, self.loss_func, self.optimizer)

        return train_loss, train_accuracy, train_time

    def eval(self, split, eval_loader):
        eval_loss, eval_accuracy, eval_time = load_one_epoch(split, eval_loader, self.model, self.loss_func, self.optimizer)

        if split == 'val':
            self.scheduler.step()

        return eval_loss, eval_accuracy, eval_time



def vgg11(args, in_size, num_class, batch_normalize=True):
    features = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGGManager(features, in_size, num_class, batch_normalize, args.gpu, args.learning_rate)
    return model



def vgg13(args, in_size, num_class, batch_normalize=True):
    features = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGGManager(features, in_size, num_class, batch_normalize, args.gpu, args.learning_rate)
    return model



def vgg16(args, in_size, num_class, batch_normalize=True):
    features = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGGManager(features, in_size, num_class, batch_normalize, args.gpu, args.learning_rate)
    return model



def vgg19(args, in_size, num_class, batch_normalize=True):
    features = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model = VGGManager(features, in_size, num_class, batch_normalize, args.gpu, args.learning_rate)
    return model
