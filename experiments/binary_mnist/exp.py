import torch
import pytorch_lightning as pl

import torch.optim as optim
from torch import nn
import neural_circuits as nc

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BinarizeAndFlatten:
    def __call__(self, x):
        return (x > 0.5).float().flatten()



def preprocess_mnist():
    """
    Loads and preprocesses the MNIST dataset by binarizing the images.
    Returns train and test DataLoaders.
    """
    transform = transforms.Compose([
    transforms.ToTensor(),
    BinarizeAndFlatten()  # Binarize & Flatten to 1D
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, persistent_workers=True, multiprocessing_context="spawn", num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, persistent_workers=True, multiprocessing_context="spawn", num_workers=7)

    return train_loader, test_loader

import torch.nn.functional as F
from torchmetrics.classification import Accuracy

class MNISTTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nc.model.BinaryLogicMNIST()
        self.loss_fn = nn.CrossEntropyLoss()

        # Train & Test Accuracy Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # 로그 저장 리스트
        self.train_acc_log = []
        self.test_acc_log = []
        self.epoch_log = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Train Accuracy 업데이트
        self.train_acc.update(logits, y)
        acc = self.train_acc.compute()  # 정확도 계산

        # 정확도를 self.log로 저장
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        # 정확도 로그 리스트에 추가
        self.train_acc_log.append(float(acc))  # .item() 대신 float() 사용

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Test Accuracy 업데이트
        self.test_acc.update(logits, y)
        acc = self.test_acc.compute()

        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.test_acc_log.append(float(acc))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

    def on_train_epoch_end(self):
        # Epoch 기록
        self.epoch_log.append(len(self.epoch_log) + 1)

        # Train Accuracy 초기화
        self.train_acc.reset()
        self.test_acc.reset()

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_accuracy_from_csv(logger):
    log_dir = logger.log_dir  # 로그 파일이 저장된 디렉토리
    csv_path = os.path.join(log_dir, "metrics.csv")  # Lightning의 CSV 로깅 경로

    if not os.path.exists(csv_path):
        print(f"⚠ No CSV log file found at {csv_path}! Check if logging is enabled.")
        return

    # CSV 파일 불러오기
    df = pd.read_csv(csv_path)

    if 'train_acc_epoch' not in df.columns or 'test_acc' not in df.columns:
        print("⚠ 'train_acc_epoch' 또는 'test_acc' 컬럼이 CSV 로그에 존재하지 않습니다.")
        print("로그된 컬럼 목록:", df.columns)
        return

    # 정확도 데이터 가져오기
    epochs = df["epoch"]
    train_acc = df["train_acc_epoch"]
    test_acc = df["test_acc"]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_acc, label="Test Accuracy", marker='s')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Test Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = preprocess_mnist()

    from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger("logs", name="binary_mnist")

    # Initialize model
    model = MNISTTrainer()

    # Training
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator=device,
        logger=logger
    )
    trainer.fit(model, train_loader, test_loader)

    # CSV Logger를 활용한 정확도 시각화
    plot_accuracy_from_csv(logger)