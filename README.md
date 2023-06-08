# PosTeQ: Position-aware Textural Quantization for Fine-grained Image Classification


## 1. Overview
This repository provides the source code of PosTeQ.

We propose two quantization methods:  
1. Position-aware Textural Quantization (PosTeQ) that exploits multiple bit-precisions while using position information.
2. Minimum Textural Quantization (MinTeQ) that exploits multiple bit-precisions while minimizing the quantization error.


## 2. Main Results

### 2-1. Weight Comparison between Fine and Coarse-grained Image Classifier
- Weight distribution of fine-grained classifier (e.g., B-CNN) shows high variance on fine-grained dataset (e.g., CUB200)
- Weight of fine-grained classifier shows lower value in deeper convolutional layer and middle output channel
![B-CNN_CUB200_histogram](https://github.com/woo-joo/PosTeQ/assets/71494469/48a8b037-12fc-4838-b655-c4f835942fbb)
![B-CNN_CUB200_heatmap](https://github.com/woo-joo/PosTeQ/assets/71494469/1ce3bc77-716f-4aaa-b307-62911eead553)

### 2-2. Quantizer Performance Comparison
- PosTeQ performs better than MinTeQ on fine-grained classifier while performs worse on coarse-grained classifier
- PosTeQ performs much better on fine-grained classifier than on coarse-grained classifier
![VGG19_CIFAR100_accuracy](https://github.com/woo-joo/PosTeQ/assets/71494469/e36c79f4-660e-41ff-9590-fcab86946683)
![B-CNN_CUB200_accuracy](https://github.com/woo-joo/PosTeQ/assets/71494469/a2f147ba-1362-4d7f-8e07-daf626d797b3)


## 3. Requirements
- Python version: 3.10.11
- Pytorch version: 1.12.1
- Torchvision version: 0.13.1


## 4. Usage
You need to specify the model and the dataset
```
python3 train.py --model B-CNN --dataset CUB200 --learning_rate 1.0 --gpu
python3 test.py --model B-CNN --dataset CUB200 --gpu
python3 plot.py --model B-CNN --dataset CUB200
```
