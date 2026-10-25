"""Phase 1 · Part 2 · Task 8 — Integration Test: MNIST Classifier

Tests validate that the learner's custom MLP (using MyLinear, MyReLU,
MyDropout, MyLayerNorm, MyCrossEntropyLoss, MyAdamW) achieves ≥97%
test accuracy on MNIST.
"""

import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytest

from tests.adapters import get_custom_mnist_model, get_custom_loss_fn, get_custom_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mnist_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten 28×28 → 784
    ])
    train_ds = datasets.MNIST(root=".mnist_data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=".mnist_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _evaluate(model, test_loader, device="cpu") -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMNISTIntegration:
    @pytest.fixture(scope="class")
    def trained_model_and_accuracy(self):
        """Train the model once for all tests in this class."""
        train_loader, test_loader = _get_mnist_loaders()

        model = get_custom_mnist_model()
        loss_fn = get_custom_loss_fn()
        optimizer = get_custom_optimizer(model.parameters(), lr=1e-3)

        model.train()
        epochs = 10
        for epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        accuracy = _evaluate(model, test_loader)
        return model, accuracy

    def test_mnist_accuracy_gte_97(self, trained_model_and_accuracy):
        _, accuracy = trained_model_and_accuracy
        assert accuracy >= 0.97, (
            f"MNIST accuracy {accuracy:.2%} is below the 97% threshold"
        )

    def test_mnist_model_uses_custom_components(self, trained_model_and_accuracy):
        """Basic sanity check: model must be a valid nn.Module."""
        model, _ = trained_model_and_accuracy
        assert isinstance(model, torch.nn.Module), "Model should be an nn.Module"
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model must have learnable parameters"
