from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pyplot


def plot_data(plot, x, y):
    plot.plot(x[y == 0, 0], x[y == 0, 1], 'ob', alpha=0.5)
    plot.plot(x[y == 1, 0], x[y == 1, 1], 'xr', alpha=0.5)
    plot.legend(['Cluster 0', 'Cluster 1'])
    return plot


X, y = make_circles(n_samples=1000, factor=0.6, noise=0.1, random_state=64)

plot = plot_data(pyplot, X, y)
plot.show()


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return torch.sigmoid(x)


Net = NeuralNetwork()
print(Net)

criterion = nn.BCELoss()
optimizer = optim.Adam(Net.parameters(), lr=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

for epoch in range(60):
    optimizer.zero_grad()
    loss = 0
    for i in range(len(X_train)):
        input = Variable(torch.FloatTensor(X_train[i]))
        if y_train[i] == 1:
            ground_truth = Variable(torch.ones(1))
        else:
            ground_truth = Variable(torch.zeros(1))

        pred = Net(input)
        loss += criterion(pred, ground_truth)

    loss.backward()
    optimizer.step()
    print("For epoch %d we get a loss of: %f" % ((epoch + 1), loss / len(X_train)))
