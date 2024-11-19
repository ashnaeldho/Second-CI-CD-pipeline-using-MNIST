# MNIST CNN Training Pipeline

This project implements a CNN model for MNIST digit classification with data augmentation, training, and visualization capabilities.

## Features

- Convolutional Neural Network for MNIST digit classification
- Data augmentation pipeline with:
  - Random affine transformations
  - Random rotation
  - Brightness/contrast adjustments
- Training pipeline with OneCycleLR scheduler
- CI/CD pipeline with automated testing
- Augmentation visualization tool
- Comprehensive test suite including:
  - Model parameter count verification (<24,000 parameters)
  - Model accuracy verification (>96% on test set)
  - Precision and recall metrics (>95%)
  - Loss value verification (<1.0)
  - Output shape validation (10 classes)

## Requirements

Install dependencies using:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Train the model**:
```bash
python train.py
```

2. **Visualize augmentations**:
```bash
python visualize_augmentations.py
```
This will generate augmented versions of all digits (0-9) in `visualizations/augmented_digits.png`

3. **Run tests**:
```bash
pytest ci_cd_pipeline.py -v
```

## Model Architecture
See HOW-TO.md for detailed architecture information.

## CI/CD Pipeline
The project includes GitHub Actions for:
- Code linting (flake8, black)
- Automated testing including:
  - Model output validation
  - Loss verification
  - Precision/recall metrics
  - Parameter count checks
- Model training
- Model validation
- Artifact storage 

## Project Structure
```
.
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── models/              # Created automatically
├── visualizations/      # Created automatically
├── data/               # Created automatically
├── .gitignore
├── README.md
├── HOW-TO.md
├── requirements.txt
├── mnist_cnn.py
├── train.py
├── visualize_augmentations.py
└── ci_cd_pipeline.py
```

## Test Suite Details

The test suite verifies:
1. Model Architecture:
   - Correct output dimensions (10 classes)
   - Parameter count < 24,000

2. Model Performance:
   - Loss < 1.0 on test set
   - Precision > 95%
   - Recall > 95%
   - Accuracy > 96%

3. Model Deployment:
   - Successful TorchScript conversion
   - Artifact generation

## Artifacts
The trained models will be saved with timestamps and device info (e.g., mnist_model_20230615_123456_cuda.pth)
Visualizations will be saved in the visualizations directory