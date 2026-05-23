
import torch

class HighPassFilter(torch.nn.Module):

    def __init__(self, kernel):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(kernel + 1, 1, kernel // 2)

    def forward(self, x):
        return x - self.pool(x)

def Conv(c_in, c_out, kernel):
    return torch.nn.Sequential(
        HighPassFilter(kernel),
        torch.nn.Conv1d(c_in, c_out, 3, 1, 1),
        torch.nn.SiLU())

class Residual(torch.nn.Module):

    def __init__(self, c, k, n=3):
        super().__init__()
        self.layers = torch.nn.Sequential(*[Conv(c, c, k) for _ in range(n)])
        
    def forward(self, x):
        return self.layers(x) + x

def Down(c_in, c_out, kernel):
    return torch.nn.Sequential(
        Residual(c_in, kernel), 
        Conv(c_in, c_out, kernel),
        torch.nn.MaxPool1d(2))

def Up(c_in, c_out, kernel):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2),
        Conv(c_in, c_out, kernel),
        Residual(c_out, kernel))
