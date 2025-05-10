import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def generate_datas(batch_size):
    train_dataset_dir = "/workspace/dataset"
    mnist_train = datasets.FashionMNIST(root=train_dataset_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset_dir = "/workspace/dataset"
    mnist_test = datasets.FashionMNIST(root=test_dataset_dir, train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader

class SoftmaxRegression(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self._linear = nn.Linear(num_inputs, num_outputs)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        return self._softmax(self._linear(inputs))

class MetricsPlotter:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.metrics = []

    def add(self, epoch, loss, metric):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.metrics.append(metric)

    def show(self, file):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.metrics, label='Test Metric (Accuracy)')
        plt.xlabel('Epoch')
        plt.ylabel('Metric (Accuracy)')
        plt.title('Test Metric over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig(file)

if __name__ == "__main__":
    num_epoch = 10
    batch_size = 256

    train_dataloader, test_dataloader = generate_datas(batch_size)
    # picture: (1, 28 ,28), class_num: 10
    model = SoftmaxRegression(28 * 28, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    plotter = MetricsPlotter()
    for epoch in range(num_epoch):
        print(f"########### epoch{epoch} ###########")
        # train
        model.train()
        losses = []
        for input, target in train_dataloader:
            pred = model(input)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f"loss: {sum(losses) / len(losses)}")
        # test
        model.eval()
        with torch.no_grad():
            metrics = []
            for input, target in test_dataloader:
                pred = model(input)
                metric = (pred.argmax(dim=1) == target).float().mean()
                metrics.append(metric.item())
            print(f"metric: {sum(metrics) / len(metrics)}")
        plotter.add(epoch, sum(losses) / len(losses), sum(metrics) / len(metrics))
    plotter.show("./viz.png")