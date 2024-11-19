# MNIST CNN Model

A lightweight CNN implementation for MNIST digit classification that achieves 96%+ accuracy in one epoch with less than 24,000 parameters.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Testing

```bash
pytest ci_cd_pipeline.py -v
```

## Model Architecture
See HOW-TO.md for detailed architecture information.

## CI/CD Pipeline
The project includes GitHub Actions for:
- Code linting (flake8, black)
- Automated testing
- Model training
- Model validation
- Artifact storage 


Directory structure should look like:
.
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
├── models/              # Created automatically
├── data/               # Created automatically
├── .gitignore
├── README.md
├── HOW-TO.md
├── requirements.txt
├── mnist_cnn.py
├── train.py
└── ci_cd_pipeline.py

After setting up these files:
Initialize git repository
Make initial commit
Create GitHub repository
Push to GitHub
GitHub Actions will automatically run the pipeline
The trained models will be saved with timestamps and device info (e.g., mnist_model_20230615_123456_cuda.pth), making it easy to track when and where they were trained.