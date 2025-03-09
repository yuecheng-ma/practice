import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def save_images(imgs, num_rows, num_cols, titles=None, scale=1.5, save_path='output.png'):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def load_data_fashion_mnist(batch_size):
    train_dataset_dir = "/workspace/deepleaning/dataset"
    mnist_train = torchvision.datasets.FashionMNIST(root=train_dataset_dir, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset_dir = "/workspace/deepleaning/dataset"
    mnist_test = torchvision.datasets.FashionMNIST(root=test_dataset_dir, train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader

class SoftmaxReg():
    def __init__(self, input_dim, num_classes):
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, num_classes), nn.Softmax(dim=1))

        self.loss = nn.CrossEntropyLoss()
        self.trainer = torch.optim.SGD(self.net.parameters(), lr=0.1)

    def forward(self, x):
        # x = self.flatten_layer(x)
        # logits = self.linear_layer(x)
        # probs = self.softmax_layer(logits)
        probs = self.net(x)
        return probs


def calcu_accuracy(pred: torch.Tensor, gt: torch.Tensor):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = pred.argmax(dim=1)
    cmp = pred.type(gt.dtype) == gt
    return float(cmp.type(gt.dtype).sum()) / len(gt)

if __name__ == "__main__":
    num_epoch = 50
    batch_size = 256
    train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
    # picture: (1, 28 ,28), class_num: 10
    net = SoftmaxReg(28 * 28, 10)
    for epoch in range(num_epoch):
        print(f"########### epoch{epoch} ###########")
        # train
        losses = []
        for x, y in train_dataloader:
            pred = net.forward(x)
            loss = net.loss(pred, y)
            net.trainer.zero_grad()
            loss.backward()
            net.trainer.step()
            losses.append(loss.item())
        print(f"loss: {sum(losses) / len(losses)}")
        # test
        with torch.no_grad():
            metrics = []
            for x, y in test_dataloader:
                pred = net.forward(x)
                metric = calcu_accuracy(pred, y)
                metrics.append(metric)
            print(f"metric: {sum(metrics) / len(metrics)}")
