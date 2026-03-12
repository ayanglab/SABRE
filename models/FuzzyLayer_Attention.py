import torch
import torch.nn as nn


class FuzzyLayer_AND(nn.Module):
    def __init__(self, input_channel, reduction_channel, fuzzynum):
        super(FuzzyLayer_AND, self).__init__()
        # n is the membership
        self.conv1 = nn.Conv3d(input_channel, reduction_channel, kernel_size=1)
        self.norm = nn.InstanceNorm3d(reduction_channel)
        self.act = nn.LeakyReLU(inplace=True)

        self.fuzzynum = fuzzynum
        self.channel = input_channel
        self.reduction_channel = reduction_channel
        self.mu = nn.Parameter(torch.randn((1, self.fuzzynum, self.reduction_channel,1,1,1)))
        self.sigma = nn.Parameter(torch.randn((1, self.fuzzynum, self.reduction_channel, 1,1,1)))


    def forward(self, x):
        f0 = self.conv1(x)
        f0 = self.act(self.norm(f0))
        fuzzy_and = torch.ones(f0.size(), dtype=torch.float).cuda()
        for i in range(self.fuzzynum):
            fuzzy = torch.square((f0 - self.mu[:,i,...]) / self.sigma[:,i,...])
            fuzzy_and += fuzzy
        fuzzy_and_ = torch.exp(-fuzzy_and)
        return fuzzy_and_


class FuzzyLayer_OR(nn.Module):
    def __init__(self, input_channel, reduction_channel, fuzzynum):
        super(FuzzyLayer_OR, self).__init__()
        # n is the membership
        self.conv1 = nn.Conv3d(input_channel, reduction_channel, kernel_size=1)
        self.norm = nn.InstanceNorm3d(reduction_channel)
        self.act = nn.LeakyReLU(inplace=True)

        self.fuzzynum = fuzzynum
        self.channel = input_channel
        self.reduction_channel = reduction_channel
        self.mu = nn.Parameter(torch.randn((1, self.fuzzynum, self.reduction_channel,1,1,1)))
        self.sigma = nn.Parameter(torch.randn((1, self.fuzzynum, self.reduction_channel, 1,1,1)))


    def forward(self, x):
        f0 = self.conv1(x)
        f0 = self.act(self.norm(f0))
        fuzzy_or = torch.zeros(f0.size(), dtype=torch.float).cuda()
        for i in range(self.fuzzynum):
            fuzzy = torch.square((f0 - self.mu[:,i,...]) / self.sigma[:,i,...])
            fuzzy_or = torch.maximum(fuzzy_or, fuzzy)
        fuzzy_or_ = torch.exp(-fuzzy_or)
        return fuzzy_or_



if __name__ == '__main__':
    use_gpu = True
    net = FuzzyLayer_OR(32, 32, 4)
    if use_gpu:
        net = net.cuda()
        inputs = torch.randn(2, 32, 64, 128, 128).cuda()
    else:
        inputs = torch.randn(2, 32, 64, 128, 128)

    output = net(inputs)
    print(output.shape)