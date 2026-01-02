
import torch

def Conv(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Conv1d(c_in, c_out, 3, 1, 1, bias=False),
        torch.nn.BatchNorm1d(c_out),
        torch.nn.SiLU())

class Residual(torch.nn.Module):

    def __init__(self, c, n=3):
        super().__init__()
        self.layers = torch.nn.Sequential(*[Conv(c, c) for _ in range(n)])
        
    def forward(self, x):
        return self.layers(x) + x

def Down(c_in, c_out):
    return torch.nn.Sequential(
        Residual(c_in), 
        Conv(c_in, c_out),
        torch.nn.MaxPool1d(2))

def Up(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2),
        Conv(c_in, c_out),
        Residual(c_out))
