import torch
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
    def __init__(self, in_features, out_features):
        self.w = torch.normal(0, 0.01, size=(in_features, out_features), requires_grad=True)
        self.b = torch.zeros(out_features, requires_grad=True)

    def forward(self, x):
        return self.softmax(torch.matmul(torch.flatten(x, start_dim=1), self.w) + self.b) # x.view(x.shape[0], -1)

    def softmax(self, x):
        exp_x = torch.exp(x)
        partition = exp_x.sum(1, keepdim=True)
        return exp_x / partition

    # cross_entropy loss
    def loss(self, pred, gt):
        loss = -torch.log(pred[range(len(pred)), gt])
        return loss.mean()

    def sgd(self, learning_rate, batch_size):
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad / batch_size
            self.b -= learning_rate * self.b.grad / batch_size
            self.w.grad.zero_()
            self.b.grad.zero_()

def calcu_accuracy(pred: torch.Tensor, gt: torch.Tensor):
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = pred.argmax(dim=1)
    cmp = pred.type(gt.dtype) == gt
    return float(cmp.type(gt.dtype).sum()) / len(gt)

if __name__ == "__main__":
    num_epoch = 50
    learning_rate = 0.1
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
            loss.backward()
            net.sgd(learning_rate, batch_size)
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
