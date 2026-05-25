
import torch

class MovingNorm(torch.nn.Module):

    def __init__(self, channel, size):
        super().__init__()
        std = (size - 1) / 6
        kernel = (torch.arange(size) - (size - 1) / 2) / std
        kernel = torch.exp(-0.5 * kernel.square())
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.repeat((channel, 1, 1)))
        self.pad = torch.nn.ZeroPad1d(size // 2)
        self.mean = torch.nn.Parameter(torch.randn(1, channel, 1) * 0.001)
        self.std = torch.nn.Parameter(torch.randn(1, channel, 1) * 0.001 + 1)
        self.channel = channel

    def forward(self, x):
        mean = torch.nn.functional.conv1d(self.pad(x), self.kernel, None, 1, groups=self.channel)
        delta = x - mean
        std = (torch.mean(delta.square(), (0, 2), True) + 1e-9).sqrt()
        return self.mean + delta * self.std / std

class Snake(torch.nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(1, channel, 1) * 0.001 - 2)

    def forward(self, x):
        alpha = 0.1 + torch.nn.functional.sigmoid(self.alpha) * 10
        return x + torch.sin(alpha * x).square() / alpha

def Conv(c_in, c_out, size):
    return torch.nn.Sequential(
        torch.nn.Conv1d(c_in, c_out, 3, 1, 1, bias=False),
        MovingNorm(c_out, size),
        Snake(c_out))

class Residual(torch.nn.Module):

    def __init__(self, c, s, n=3):
        super().__init__()
        self.layers = torch.nn.Sequential(*[Conv(c, c, s) for _ in range(n)])
        
    def forward(self, x):
        return self.layers(x) + x

def Down(c_in, c_out, size):
    return torch.nn.Sequential(
        Residual(c_in, size), 
        Conv(c_in, c_out, size),
        torch.nn.MaxPool1d(2))

def Up(c_in, c_out, size):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2),
        Conv(c_in, c_out, size),
        Residual(c_out, size))
