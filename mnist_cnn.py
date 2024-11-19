import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # First Convolution Block
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(12)
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.bn2 = nn.BatchNorm2d(16)
        
        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)  # 7x7 -> 7x7
        self.bn3 = nn.BatchNorm2d(20)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(20 * 3 * 3, 40)
        self.bn_fc1 = nn.BatchNorm1d(40)
        self.fc2 = nn.Linear(40, 10)
        
        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # First block with skip connection
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)  # 7x7 -> 3x3
        
        # Fully connected layers
        x = x.view(-1, 20 * 3 * 3)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_test(model, device, train_loader, test_loader, optimizer, scheduler, epoch):
    """Train and test the model for one epoch"""
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += target.size(0)
        
        # Update progress bar
        train_accuracy = 100. * train_correct / train_total
        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f} Train Acc: {train_accuracy:.2f}%')
    
    # Calculate final training metrics
    train_loss /= len(train_loader)
    train_accuracy = 100. * train_correct / train_total
    
    # Testing
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
    
    test_loss /= test_total
    test_accuracy = 100. * test_correct / test_total
    
    # Print epoch summary
    print(f'\nEpoch {epoch} Summary:')
    print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_correct}/{train_total} ({train_accuracy:.2f}%)')
    print(f'Testing  - Loss: {test_loss:.4f}, Accuracy: {test_correct}/{test_total} ({test_accuracy:.2f}%)\n')
    
    return test_accuracy