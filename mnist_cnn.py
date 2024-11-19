import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Layer 1: Input = 28x28x1, Output = 28x28x16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 2: Input = 14x14x16, Output = 14x14x20
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        
        # Layer 3: Input = 7x7x20, Output = 7x7x24
        self.conv3 = nn.Conv2d(20, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        
        # Layer 4: Input = 4x4x24, Output = 4x4x24
        self.conv4 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        
        # Layer 5: Input = 2x2x24, Output = 2x2x24
        self.conv5 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        
        # Final fully connected layer
        self.fc = nn.Linear(24, 10)  # After global average pooling

    def forward(self, x):
        # Layer 1: 28x28 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)
        
        # Layer 2: 14x14 -> 7x7
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)
        
        # Layer 3: 7x7 -> 4x4
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)
        
        # Layer 4: 4x4 -> 2x2
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)
        
        # Layer 5: 2x2 -> 1x1
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    """Count and print trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nModel Parameter Analysis:')
    print('=' * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name:30} : {param.numel():,} parameters')
    print('=' * 50)
    print(f'Total Trainable Parameters: {total_params:,}\n')
    return total_params

def train_and_test(model, device, train_loader, test_loader, optimizer, scheduler, epoch):
    # Training
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f'\nEpoch {epoch}:')
    print('-' * 30)
    
    with tqdm(train_loader, desc='Training') as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/len(train_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    train_loss = total_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    
    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{test_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
    
    test_loss /= total
    test_accuracy = 100. * correct / total
    
    # Print epoch results
    print(f'\nResults for Epoch {epoch}:')
    print(f'Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
    print(f'Testing  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%\n')
    
    return test_accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set hyperparameters
    BATCH_SIZE = 64  # Adjusted batch size
    EPOCHS = 1
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders with fixed batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False
    )
    
    # Initialize model and move to device
    model = MNIST_CNN().to(device)
    
    # Count and print parameters
    total_params = count_parameters(model)
    assert total_params < 24000, f"Model has {total_params:,} parameters, exceeding limit of 24,000"
    
    # Initialize optimizer with modified parameters
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,  # Adjusted learning rate
        momentum=0.9,
        weight_decay=5e-4,  # Increased weight decay
        nesterov=True
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        div_factor=10
    )
    
    # Modify train_and_test function to use scheduler
    for epoch in range(1, EPOCHS + 1):
        accuracy = train_and_test(model, device, train_loader, test_loader, optimizer, scheduler, epoch)
        assert accuracy >= 96.0, f"Model accuracy {accuracy:.2f}% is below required 96%"

if __name__ == '__main__':
    main() 