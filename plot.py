import argparse
import os.path
import matplotlib.pyplot as plt



def read_train_log(train_log_file):
    logs = train_log_file.readlines()
    losses, accuracies = {'train': [], 'test': []}, {'train': [], 'test': []}
    for i in range(len(logs)):
        split = 'train' if int(i) % 2 == 0 else 'test'
        _, loss, accuracy = logs[i].split(',')
        losses[split].append(float(loss))
        accuracies[split].append(float(accuracy))
    
    return losses, accuracies



def train_plot(train, test, y_label, title, path):
    plt.figure(figsize=(4, 3))
    plt.title(f'{title}_{y_label}')

    plt.plot(range(len(train)), train, '-')
    plt.plot(range(len(test)),  test, '-')

    plt.xlabel('epoch')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(['train', 'test'])
    plt.savefig(os.path.join(path, f'{title}_train_{y_label}.png'))



def read_test_log(test_log_file):
    logs = test_log_file.readlines()
    base_accuracy = float(logs[0].split(',')[2])
    quantizers, precisions, accuracies = [], [], {}
    for log in logs[1:]:
        quantizer, precision, accuracy = log.split(',')

        if quantizer not in quantizers:
            quantizers.append(quantizer)
            accuracies[quantizer] = {}
        if precision not in precisions:
            precisions.append(precision)
        accuracies[quantizer][precision] = float(accuracy)
    
    accuracies = [[accuracies[quantizer][precision] for precision in precisions] for quantizer in quantizers]
    return base_accuracy, quantizers, precisions, accuracies



def test_plot(base_accuracy, quantizers, precisions, accuracies, title, path):
    plt.figure(figsize=(4, 3))
    plt.title(f'{title}_accuracy')

    plt.axhline(base_accuracy, color='r', linestyle='--')
    for i in range(len(quantizers)):
        x = [i * 0.8 + j * len(quantizers) for j in range(len(accuracies[i]))]
        plt.bar(x, accuracies[i])

    x = [((len(quantizers) - 1) / 2) * 0.8 + i * len(quantizers) for i in range(len(precisions))]
    plt.xticks(x, precisions)
    plt.xlabel('precision')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.legend(['Base'] + quantizers)
    plt.savefig(os.path.join(path, f'{title}_test_accuracy.png'))



def read_test_all_log(test_all_log_file):
    logs = test_all_log_file.readlines()
    base_accuracy = float(logs[0].split(',')[5])
    composes, precisions, accuracies_per_quantizer = [], [], {'MinTeQ': {}, 'PosTeQ': {}}
    for log in logs[4:]:
        quantizer, precision, k, m, flipped, accuracy = log.split(',')

        k, m, flipped = int(k), int(m), flipped == 'True'
        if flipped and (k != 5 or m != 4): continue

        compose = f'k={k}, m={m}' if not flipped else 'Flipped'
        if compose not in composes:
            composes.append(compose)
        if compose not in accuracies_per_quantizer[quantizer]:
            accuracies_per_quantizer[quantizer][compose] = {}
        if precision not in precisions:
            precisions.append(precision)
        accuracies_per_quantizer[quantizer][compose][precision] = float(accuracy)

    for quantizer, accuracies in accuracies_per_quantizer.items():
        accuracies_ = [[accuracies[compose][precision] for precision in precisions] for compose in composes]
        accuracies_per_quantizer[quantizer] = accuracies_
    return base_accuracy, composes, precisions, accuracies_per_quantizer



def test_all_plot(base_accuracy, composes, precisions, accuracies, title, path):
    plt.figure(figsize=(4, 3))
    plt.title(f'{title}_accuracy')

    plt.axhline(base_accuracy, color='r', linestyle='--')
    for i in range(len(composes)):
        x = [i * 0.8 + j * len(composes) for j in range(len(accuracies[i]))]
        plt.bar(x, accuracies[i])

    x = [((len(composes) - 1) / 2) * 0.8 + i * len(composes) for i in range(len(precisions))]
    plt.xticks(x, precisions)
    plt.xlabel('precision')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.legend(composes)
    plt.savefig(os.path.join(path, f'{title}_test_accuracy.png'))



def main(args):
    plot_path = os.path.join(args.path, 'plots')
    os.makedirs(os.path.join(args.path, 'plots'), exist_ok=True)

    train_log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_train.log')
    if not os.path.isfile(train_log_path):
        print(f'No train log for {args.model} on {args.dataset}!!!')
        return
    train_log_file = open(train_log_path)
    losses, accuracies = read_train_log(train_log_file)
    train_plot(losses['train'], losses['test'], 'loss', f'{args.model}_{args.dataset}', plot_path)
    train_plot(accuracies['train'], accuracies['test'], 'accuracy', f'{args.model}_{args.dataset}', plot_path)

    test_log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_test.log')
    if not os.path.isfile(test_log_path):
        print(f'No test log for {args.model} on {args.dataset}!!!')
        return
    test_log_file = open(test_log_path)
    base_accuracy, quantizers, precisions, accuracies = read_test_log(test_log_file)
    test_plot(base_accuracy, quantizers, precisions, accuracies, f'{args.model}_{args.dataset}', plot_path)

    test_all_log_path = os.path.join(args.path, f'logs/{args.model}_{args.dataset}_test_all.log')
    if not os.path.isfile(test_all_log_path):
        print(f'No test log for {args.model} on {args.dataset}!!!')
        return
    test_all_log_file = open(test_all_log_path)
    base_accuracy, composes, precisions, accuracies_per_quantizer = read_test_all_log(test_all_log_file)
    test_all_plot(base_accuracy, composes, precisions, accuracies_per_quantizer['MinTeQ'], f'{args.model}_{args.dataset}_MinTeQ', plot_path)
    test_all_plot(base_accuracy, composes, precisions, accuracies_per_quantizer['PosTeQ'], f'{args.model}_{args.dataset}_PosTeQ', plot_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['VGG19', 'B-CNN'])
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR100', 'CUB200', 'Flowers102', 'FGVCAircraft'])
    parser.add_argument('--path', type=str, default='results')
    args = parser.parse_args()

    main(args)
