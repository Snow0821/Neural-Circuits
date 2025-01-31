import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

# ✅ Define a simple PyTorch Lightning model for MNIST classification
class SimpleLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.layer_2 = nn.Linear(128, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.layer_1(x))  # Apply ReLU activation
        x = self.layer_2(x)  # Output logits
        return x

    def training_step(self, batch, batch_idx):
        """ Defines the training step for a batch """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)  # Compute cross-entropy loss
        return loss

    def configure_optimizers(self):
        """ Sets up the optimizer for training """
        return torch.optim.Adam(self.parameters(), lr=0.001)

# ✅ Create a data loader for the MNIST dataset
def get_dataloader():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=64, shuffle=True)

# ✅ Test function to verify PyTorch Lightning setup
def test_pytorch_lightning():
    model = SimpleLightningModel()
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=True)  # Quick test run
    trainer.fit(model, get_dataloader())

# ✅ Run the test
if __name__ == "__main__":
    test_pytorch_lightning()
    print("✅ PyTorch Lightning is working correctly!")
