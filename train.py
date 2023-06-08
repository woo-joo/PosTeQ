import argparse
import os.path

import torch

from utils import get_dataloader, get_model



def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    train_loader, test_loader = get_dataloader(args, ['train', 'test'])
    model = get_model(args)


    os.makedirs(os.path.join(args.path, 'logs'), exist_ok=True)
    log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_train.log')
    mode = 'a' if os.path.isfile(log_path) and args.resume else 'w'
    log_file = open(log_path, mode)

    os.makedirs(os.path.join(args.path, 'weights'), exist_ok=True)
    weight_path = os.path.join(args.path, f'weights/{args.model}_{args.dataset}.pth')


    i, best_epoch, best_result = 0, 0, 0
    if args.resume and os.path.isfile(weight_path):
        checkpoint = torch.load(weight_path)
        model.model.load_state_dict(checkpoint['state_dict'])
        i = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_result = checkpoint['best_result']
        print('Resuming training at epoch {}...'.format(i+1))

    while i < args.num_epoch:
        train_loss, train_accuracy, train_time = model.train(train_loader)
        print('[Epoch {:>4}/{:>4}] | Train | Loss: {:>2.4f}, Accuracy: {:>3.2f}%, Consumed Time: {:>3.2f}s'.format(i+1, args.num_epoch, train_loss, train_accuracy, train_time))
        log_file.write(f'{i+1},{train_loss},{train_accuracy}\n')

        test_loss, test_accuracy, test_time = model.eval('val', test_loader)
        print('                  |  Test | Loss: {:>2.4f}, Accuracy: {:>3.2f}%, Consumed Time: {:>3.2f}s'.format(test_loss, test_accuracy, test_time))
        log_file.write(f'{i+1},{test_loss},{test_accuracy}\n')

        if best_result < test_accuracy:
            best_epoch = i
            best_result = test_accuracy
            torch.save({
                        'best_state_dict': model.model.state_dict(),
                        'best_epoch': best_epoch+1,
                        'best_result': best_result
                       }, weight_path)
        
        i += 1
    print()


    checkpoint = torch.load(weight_path)
    checkpoint['state_dict'] = model.model.state_dict()
    checkpoint['epoch'] = i
    torch.save(checkpoint, weight_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['VGG19', 'B-CNN'])
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR100', 'CUB200', 'Flowers102', 'FGVCAircraft'])
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--path', type=str, default='results')
    args = parser.parse_args()

    main(args)
