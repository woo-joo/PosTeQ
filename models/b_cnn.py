from torch import nn
from torch import optim
from torchvision.models import vgg19

from utils import load_one_epoch



class BCNN(nn.Module):
    def __init__(self, in_size, num_class):
        super().__init__()

        self.in_size = in_size

        self.quant = lambda x: x
        self.feature_extractor = nn.Sequential(*list(vgg19(weights='IMAGENET1K_V1').features.children())[:-1])
        self.classifier = nn.Linear(512 * 512, num_class)
        self.dequant = lambda x: x


    def forward(self, x):
        output = self.quant(x)
        output = self.feature_extractor(output)
        output = output.view(output.shape[0], 512, (self.in_size // 16) ** 2)
        output = output.bmm(output.transpose(1, 2)) / ((self.in_size // 16) ** 2)
        output = output.view(output.shape[0], 512 * 512)
        output = output.sign() * (output.abs() + 1e-5).sqrt()
        output = nn.functional.normalize(output)
        output = self.classifier(output)
        output = self.dequant(output)
        
        return output



class BCNNManager:
    def __init__(self, in_size, num_class, gpu, learning_rate):
        super().__init__()

        self.model = BCNN(in_size, num_class)
        if gpu:
            self.model = self.model.cuda()

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer1 = optim.SGD(self.model.feature_extractor.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-8)
        self.optimizer2 = optim.SGD(self.model.parameters(), lr=learning_rate * 0.1, momentum=0.9, weight_decay=1e-5)
        self.scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, mode='max', factor=0.1, patience=3, threshold=1e-4)
        self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, mode='max', factor=0.1, patience=3, threshold=1e-4)

        self.optimizer = self.optimizer1
        self.scheduler = self.scheduler1

        self.num_epoch = 0
        self.pretrain_epoch = 50


    def train(self, train_loader):
        train_loss, train_accuracy, train_time = load_one_epoch('train', train_loader, self.model, self.loss_func, self.optimizer)

        return train_loss, train_accuracy, train_time

    def eval(self, split, eval_loader):
        eval_loss, eval_accuracy, eval_time = load_one_epoch(split, eval_loader, self.model, self.loss_func, self.optimizer)

        if split == 'val':
            self.scheduler.step(eval_accuracy)
            self.num_epoch += 1
            if self.num_epoch == self.pretrain_epoch:
                self.optimizer = self.optimizer2
                self.scheduler = self.scheduler2

        return eval_loss, eval_accuracy, eval_time



def b_cnn(args, in_size, num_class):
    model = BCNNManager(in_size, num_class, args.gpu, args.learning_rate)
    return model
