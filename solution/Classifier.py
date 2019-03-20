import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # TODO: you can add new layers to the model and play with different activation functions.
        self.hidden_size = hidden_size
        self.combined_size = hidden_size + input_size
        self.decrease_size = hidden_size

        self.relu = nn.ReLU()

        self.reduction_layer = nn.Linear(self.combined_size, self.decrease_size)
        self.i2h = nn.Linear(self.decrease_size, hidden_size)
        self.i2o = nn.Linear(self.decrease_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # TODO: If you add new layers you have to include the steps here.
        combined = self.relu(combined)
        combined = self.reduction_layer(combined)

        output = self.i2o(combined)
        hidden = self.i2h(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
