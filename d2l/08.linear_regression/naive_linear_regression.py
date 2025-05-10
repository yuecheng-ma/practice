import torch
import random

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape) # add noise
    return x, y.unsqueeze(1)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

class LinearReg():
    def __init__(self):
        self.w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def loss(self, gt, pred):
        return torch.mean((gt - pred) ** 2)

    def sgd(self, learning_rate, batch_size):
        with torch.no_grad():
            self.w.grad.zero_()
            self.b.grad.zero_()
            self.w -= learning_rate * self.w.grad / batch_size
            self.b -= learning_rate * self.b.grad / batch_size

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    learning_rate = 0.03
    num_epochs = 10
    net = LinearReg()
    batch_size = 10
    for epoch in range(num_epochs):
        for feature, label in data_iter(batch_size, features, labels):
            loss = net.loss(label, net.forward(feature))
            loss.backward()
            net.sgd(learning_rate, batch_size)
        with torch.no_grad():
            test_loss = net.loss(labels, net.forward(features))
            print(f'epoch {epoch + 1}, loss {float(test_loss)}')

