import torch
import torch.nn as nn


class SpecialAddLayer(torch.nn.Module):
    def __init__(self, sum_equal_one=True):
        """
        if sum_equal_one:
            output = a*x + b*y
            and a + b = 1
        else:
            output = x + b*y
            and b is in the interval [0, 1]
        """
        super(SpecialAddLayer, self).__init__()
        self.sum_equal_one = sum_equal_one
        self.weight = nn.Parameter(torch.ones(2))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(0)

    def forward(self, x, y):
        softmax = nn.Softmax(0)
        weight = softmax(self.weight)
        if self.sum_equal_one:
            return weight[0]*x + weight[1]*y
        else:
            return x + weight[1] * y


if __name__ == '__main__':
    net = SpecialAddLayer(sum_equal_one=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(input, target)
    loss1 = criterion(input, target)
    aux_loss = criterion(input, target)
    # print(net)
    print(net(loss1, aux_loss))
