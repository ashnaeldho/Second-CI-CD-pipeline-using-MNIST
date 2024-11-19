import os
from datetime import datetime
from mnist_cnn import MNIST_CNN, train_and_test, count_parameters
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import random
import numpy as np

# Add normalization back to transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Test transform also without normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

def train():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_suffix = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 1
    
    # Create training dataset with simple transform
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                transform=train_transform)
    
    # Test dataset with simple transform
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training dataset size: {len(train_dataset):,} images")
    print(f"Testing dataset size: {len(test_dataset):,} images")
    
    # Initialize model
    model = MNIST_CNN().to(device)
    
    # Verify parameter count
    total_params = count_parameters(model)
    assert total_params < 24000, f"Model has {total_params:,} parameters, exceeding limit of 24,000"
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        pct_start=0.2
    )
    
    # Train model
    best_accuracy = 0
    for epoch in range(1, EPOCHS + 1):
        accuracy = train_and_test(model, device, train_loader, test_loader, optimizer, scheduler, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            best_model_path = f'models/mnist_model_best_{device_suffix}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    
    assert best_accuracy >= 96.0, f"Best model accuracy {best_accuracy:.2f}% is below required 96%"
    
    # Save final model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/mnist_model_{timestamp}_{device_suffix}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nFinal model saved to {model_path}")
    print(f"Best accuracy achieved: {best_accuracy:.2f}%")

if __name__ == '__main__':
    train() 