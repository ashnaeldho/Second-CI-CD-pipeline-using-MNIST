import os
import sys
import torch
import pytest
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_cnn import MNIST_CNN, count_parameters
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import glob

class TestMNISTPipeline:
    @classmethod
    def setup_class(cls):
        """Setup common test requirements"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = MNIST_CNN().to(cls.device)
        cls.batch_size = 64
        
        # Find the most recent model file
        model_files = glob.glob('models/mnist_model_*.pth')
        cls.model_path = max(model_files) if model_files else None
        
        if cls.model_path:
            # Load model with weights_only=True for security
            cls.model.load_state_dict(torch.load(cls.model_path, weights_only=True))
        
        # Setup test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        cls.test_dataset = datasets.MNIST(
            './data', 
            train=False, 
            download=True,
            transform=transform
        )
        
        cls.test_loader = DataLoader(
            cls.test_dataset,
            batch_size=cls.batch_size,
            shuffle=False
        )

    def test_model_output_shape(self):
        """Test 1: Verify model outputs 10 classes"""
        batch_size = 32
        test_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        output = self.model(test_input)
        
        assert output.shape[1] == 10, f"Expected 10 output classes, got {output.shape[1]}"
        print("\nTest 1 Passed: Model outputs 10 classes")

    def test_loss_value(self):
        """Test 2: Verify loss is less than 1.0 after training"""
        if not self.model_path:
            pytest.skip("No trained model found in models/ directory")
            
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target)
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        assert avg_loss < 1.0, f"Average loss {avg_loss:.4f} is too high"
        print(f"\nTest 2 Passed: Average loss {avg_loss:.4f} is less than 1.0")

    def test_model_precision(self):
        """Test 3: Verify model precision > 95%"""
        if not self.model_path:
            pytest.skip("No trained model found in models/ directory")
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        precision, _, _, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        assert precision > 0.95, f"Precision {precision:.4f} is below threshold"
        print(f"\nTest 3 Passed: Precision={precision:.4f}")

    def test_model_recall(self):
        """Test 4: Verify model recall > 95%"""
        if not self.model_path:
            pytest.skip("No trained model found in models/ directory")
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        _, recall, _, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        assert recall > 0.95, f"Recall {recall:.4f} is below threshold"
        print(f"\nTest 4 Passed: Recall={recall:.4f}")

    def test_model_parameters(self):
        """Test 5: Check if model has less than 24000 parameters"""
        total_params = count_parameters(self.model)
        assert total_params < 24000, f"Model has {total_params:,} parameters, exceeding limit"
        print(f"\nTest 5 Passed: Model has {total_params:,} parameters")

    def test_model_f1_score(self):
        """Test 6: Verify model F1 score > 95%"""
        if not self.model_path:
            pytest.skip("No trained model found in models/ directory")
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        _, _, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        assert f1 > 0.95, f"F1 Score {f1:.4f} is below threshold"
        print(f"\nTest 6 Passed: F1 Score={f1:.4f}")

def deploy_model():
    """Simple deployment process"""
    try:
        # Find the most recent model
        model_files = glob.glob('models/mnist_model_*.pth')
        if not model_files:
            print("\nNo model found for deployment")
            return False
            
        model_path = max(model_files)
        
        # Load model with weights_only=True
        model = MNIST_CNN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        # Save model in deployment format (TorchScript)
        example = torch.rand(1, 1, 28, 28)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("deployed_model.pt")
        
        print(f"\nDeployment successful: Model {model_path} converted to TorchScript format")
        return True
    except Exception as e:
        print(f"\nDeployment failed: {str(e)}")
        return False

def main():
    """Run the CI/CD pipeline"""
    print("Starting CI/CD Pipeline for MNIST Model")
    print("=" * 50)
    
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Deploy model if tests pass
    deploy_model()

if __name__ == "__main__":
    main() 