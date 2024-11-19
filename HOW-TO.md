# MNIST CNN Implementation Guide

This guide explains how to use and understand the lightweight CNN implementation for MNIST digit classification that achieves 96%+ accuracy in one epoch.

## Prerequisites

- Python 3.6+
- PyTorch
- torchvision
- tqdm
- CUDA-capable GPU (optional, but recommended)

## Installation 

```bash
pip install torch torchvision tqdm

```

## Model Architecture

The model is a lightweight 5-layer CNN with less than 24,000 parameters:

1. Conv1: 1 → 8 channels (3x3 kernel, padding=1)
2. Conv2: 8 → 16 channels (3x3 kernel, padding=1)
3. Conv3: 16 → 16 channels (3x3 kernel, padding=1)
4. Conv4: 16 → 20 channels (3x3 kernel, padding=1)
5. Conv5: 20 → 20 channels (3x3 kernel, padding=1)
- Final: Fully connected layer (80 → 10)

Each convolutional layer is followed by:
- ReLU activation
- Max Pooling (2x2) (except last layer)

## Key Features

- Efficient architecture with gradual increase in filters
- Kaiming initialization for weights
- Max pooling for dimension reduction
- Single fully connected layer
- Uses Adam optimizer
- Expected accuracy: >96% after 1 epoch

## Running the Code

1. Save both files in the same directory:
   - `mnist_cnn.py`
   - `HOW-TO.md`

2. Run the training script:
   ```bash
   python mnist_cnn.py
   ```

3. The script will:
   - Show which device (CPU/CUDA) is being used
   - Display detailed parameter count for each layer
   - Download MNIST dataset (first run only)
   - Show training progress with live loss and accuracy
   - Show testing progress with live loss and accuracy
   - Display final results for the epoch

## Expected Output

You should see something like:
```
Using device: cuda

Model Parameter Analysis:
==================================================
conv1.weight                     : 72 parameters
conv1.bias                       : 8 parameters
conv2.weight                     : 1,152 parameters
conv2.bias                       : 16 parameters
conv3.weight                     : 2,304 parameters
conv3.bias                       : 16 parameters
conv4.weight                     : 2,880 parameters
conv4.bias                       : 20 parameters
conv5.weight                     : 3,600 parameters
conv5.bias                       : 20 parameters
fc.weight                        : 800 parameters
fc.bias                          : 10 parameters
==================================================
Total Trainable Parameters: 10,898

Epoch 1:
------------------------------
Training: 100%|██████████| 938/938 [00:XX<00:00] loss: 0.XXXX, acc: XX.XX%
Testing: 100%|██████████| 157/157 [00:XX<00:00] loss: 0.XXXX, acc: 96.XX%

Results for Epoch 1:
Training - Loss: 0.XXXX, Accuracy: XX.XX%
Testing  - Loss: 0.XXXX, Accuracy: 96.XX%
```

## Model Parameters

- Input: 28x28 grayscale images
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Negative Log Likelihood
- Weight initialization: Kaiming normal for conv layers, Normal(0, 0.01) for FC layer

## Troubleshooting

1. If you get dataset download errors:
   - Check your internet connection
   - Try using a VPN if needed
   - Manually download from http://yann.lecun.com/exdb/mnist/
   - Place files in ./data/MNIST/raw/

2. If you get CUDA out of memory error:
   - Reduce batch size (currently 64)
   - Run on CPU instead

3. If accuracy is below 96%:
   - Check if data normalization is correct (mean=0.1307, std=0.3081)
   - Verify weight initialization
   - Try adjusting learning rate
   - Ensure all layers are properly connected

## Performance Notes

- GPU training is significantly faster
- First epoch should achieve >96% accuracy
- Model uses approximately 11,000 parameters
- Architecture is optimized for:
  * Fast training
  * High accuracy
  * Minimal parameter count
  * Efficient feature extraction
- Assertions ensure:
  * Parameter count < 24,000
  * Accuracy >= 96%