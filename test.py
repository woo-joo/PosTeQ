import argparse
import os.path

import torch

from utils import get_dataloader, get_model, quantize_model



def test(model, state_dict, test_loader, quantizer, precision, k, m, flipped, all_log_file):
    model.model.load_state_dict(state_dict)
    quantize_model(model.model, quantizer, precision, k, m, flipped)

    _, test_accuracy, test_time = model.eval('test', test_loader)
    print('Quantizer: {:>7}, Precision: {}, k: {}, m: {}, flipped: {:>5} | Accuracy: {:>3.2f}%, Consumed Time: {:>3.2f}s\n'.format(quantizer, precision, k, m, str(flipped), test_accuracy, test_time))
    all_log_file.write(f'{quantizer},{precision},{k},{m},{str(flipped)},{round(test_accuracy, 3)}\n')

    return round(test_accuracy, 3)



def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    test_loader = get_dataloader(args, ['test'])[0]
    model = get_model(args)


    os.makedirs(os.path.join(args.path, 'logs'), exist_ok=True)
    log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_test.log')
    log_file = open(log_path, 'w')
    all_log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_test_all.log')
    all_log_file = open(all_log_path, 'w')

    weight_path = os.path.join(args.path, f'weights/{args.model}_{args.dataset}.pth')
    if not os.path.isfile(weight_path):
        print(f'No trained {args.model} on {args.dataset}!!!')
        return
    state_dict = torch.load(weight_path)['best_state_dict']

    for quantizer in ['Base', 'Uniform', 'MinTeQ', 'PosTeQ']:
        if quantizer == 'Base':
            test_accuracy = test(model, state_dict, test_loader, quantizer, '-', '-', '-', '-', all_log_file)
            log_file.write(f'{quantizer},-,{test_accuracy}\n')
            continue
        for precision in [8, 6, 4]:
            if quantizer == 'Uniform':
                test_accuracy = test(model, state_dict, test_loader, quantizer, precision, '-', '-', '-', all_log_file)
                log_file.write(f'{quantizer},{precision},{test_accuracy}\n')
                continue
            test_accuracy = 0
            for k in [3, 5]:
                for m in [2, 4]:
                    for flipped in [False, True]:
                        test_accuracy = max(test_accuracy, test(model, state_dict, test_loader, quantizer, precision, k, m, flipped, all_log_file))
            log_file.write(f'{quantizer},{precision},{test_accuracy}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['VGG19', 'B-CNN'])
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR100', 'CUB200', 'Flowers102', 'FGVCAircraft'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--path', type=str, default='results')
    args = parser.parse_args()

    main(args)
