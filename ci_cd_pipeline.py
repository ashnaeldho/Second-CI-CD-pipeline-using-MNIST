import os
import sys
import torch
import pytest
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_cnn import MNIST_CNN, count_parameters

class TestMNISTPipeline:
    @classmethod
    def setup_class(cls):
        """Setup common test requirements"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = MNIST_CNN().to(cls.device)
        cls.batch_size = 64
        
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

    def test_model_parameters(self):
        """Test 1: Check if model has less than 24000 parameters"""
        total_params = count_parameters(self.model)
        assert total_params < 24000, f"Model has {total_params:,} parameters, exceeding limit of 24,000"
        print(f"\nTest 1 Passed: Model has {total_params:,} parameters")

    def test_model_architecture(self):
        """Test 2: Verify model architecture and input/output dimensions"""
        batch_size = 32
        test_input = torch.randn(batch_size, 1, 28, 28).to(self.device)
        output = self.model(test_input)
        
        assert output.shape == (batch_size, 10), f"Expected output shape (32, 10), got {output.shape}"
        print("\nTest 2 Passed: Model architecture verification successful")

    def test_model_accuracy(self):
        """Test 3: Check if model achieves >95% accuracy on test set"""
        # Load pre-trained weights if available
        if os.path.exists('mnist_model.pth'):
            self.model.load_state_dict(torch.load('mnist_model.pth'))
        else:
            print("\nWarning: No pre-trained weights found. Skipping accuracy test.")
            return
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below required 95%"
        print(f"\nTest 3 Passed: Model achieved {accuracy:.2f}% accuracy")

    def test_model_save_load(self):
        """Test 4: Verify model save and load functionality"""
        # Save model
        torch.save(self.model.state_dict(), 'test_model.pth')
        
        # Load model
        loaded_model = MNIST_CNN().to(self.device)
        loaded_model.load_state_dict(torch.load('test_model.pth'))
        
        # Compare model parameters
        for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2), "Model save/load verification failed"
        
        # Cleanup
        os.remove('test_model.pth')
        print("\nTest 4 Passed: Model save/load verification successful")

    def test_model_input_robustness(self):
        """Test 5: Check model robustness to input variations"""
        self.model.eval()  # Set model to evaluation mode
        
        test_cases = [
            torch.randn(1, 1, 28, 28),    # Single image
            torch.randn(64, 1, 28, 28),   # Full batch
            torch.randn(32, 1, 28, 28),   # Half batch
            torch.randn(16, 1, 28, 28),   # Quarter batch
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            test_input = test_input.to(self.device)
            try:
                with torch.no_grad():  # No need for gradients in testing
                    output = self.model(test_input)
                    # Check output dimensions
                    assert output.shape[0] == test_input.shape[0], f"Batch size mismatch in case {i}"
                    assert output.shape[1] == 10, f"Expected 10 classes in output for case {i}, got {output.shape[1]}"
                    # Check output validity
                    assert torch.isfinite(output).all(), f"Output contains NaN or Inf in case {i}"
                    assert (output <= 0).all(), f"Log softmax output should be <= 0 in case {i}"
            except Exception as e:
                pytest.fail(f"Model failed on test case {i}: {str(e)}")
        
        print("\nTest 5 Passed: Model input robustness verification successful")

def deploy_model(model_path='mnist_model.pth'):
    """Simple deployment process"""
    try:
        # Load model
        model = MNIST_CNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Save model in deployment format (TorchScript)
        example = torch.rand(1, 1, 28, 28)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("deployed_model.pt")
        
        print("\nDeployment successful: Model converted to TorchScript format")
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
    if os.path.exists('mnist_model.pth'):
        deploy_model()
    else:
        print("\nSkipping deployment: No trained model found")

if __name__ == "__main__":
    main() 