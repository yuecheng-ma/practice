import torch
from torch import nn
from torch.utils import data

def synthetic_data(w, b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape) # add noise
    return x, y.unsqueeze(1)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class LinearReg():
    def __init__(self):
        self.net = nn.Linear(2, 1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

        self.loss = nn.MSELoss()
        self.trainer = torch.optim.SGD(self.net.parameters(), lr=0.03)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    learning_rate = 0.03
    num_epochs = 10
    net = LinearReg()
    batch_size = 10
    for epoch in range(num_epochs):
        for feature, label in load_array((features, labels), batch_size):
            loss = net.loss(label, net.net(feature))
            net.trainer.zero_grad()
            loss.backward()
            net.trainer.step()
        with torch.no_grad():
            test_loss = net.loss(labels, net.forward(features))
            print(f'epoch {epoch + 1}, loss {float(test_loss)}')

