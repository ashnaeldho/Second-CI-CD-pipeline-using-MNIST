import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from train import train_transform
import os

def show_augmented_images():
    # Create directory for saving the visualization
    os.makedirs('visualizations', exist_ok=True)
    
    # Load MNIST dataset
    dataset = datasets.MNIST('./data', train=True, download=True, 
                           transform=transforms.ToTensor())
    
    # Create a figure to display augmented images
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.suptitle('10 Augmented Versions of Each Digit (0-9)', fontsize=16)
    
    # Get one example of each digit (0-9)
    digit_images = {}
    for image, label in dataset:
        if label not in digit_images:
            digit_images[label] = image
        if len(digit_images) == 10:
            break
    
    # Apply augmentation for each digit
    for digit in range(10):  # 0-9 digits
        image = digit_images[digit]
        # Create 10 augmented versions of each digit
        for j in range(10):
            # Convert to PIL Image for transforms
            pil_image = transforms.ToPILImage()(image)
            # Apply augmentation
            augmented = train_transform(pil_image)
            # Convert back to numpy for display
            augmented_np = augmented.squeeze().numpy()
            
            # Display the image
            axes[digit, j].imshow(augmented_np, cmap='gray')
            if j == 0:  # Add digit label on the left
                axes[digit, j].set_ylabel(f'Digit {digit}', fontsize=12)
            axes[digit, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations/augmented_digits.png')
    plt.close()
    
    print("Augmented images have been saved to 'visualizations/augmented_digits.png'")

if __name__ == '__main__':
    show_augmented_images() 