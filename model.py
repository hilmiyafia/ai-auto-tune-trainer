
import torch
from blocks import Conv, Down, Up, Residual

class Reflow(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        counts = [32, 32, 32, 64, 64, 128, 128]
        positions = torch.pow(10000, -torch.arange(8) / 8)[None, :, None]
        self.register_buffer("positions", positions)
        self.downs = torch.nn.ModuleList([Conv(5 + 16, 32)])
        for i in range(len(counts) - 1):
            self.downs.append(Down(counts[i], counts[i + 1]))
        self.ups = torch.nn.ModuleList([Residual(128)])
        for i in reversed(range(len(counts) - 1)):
            self.ups.append(Up(2 * counts[i + 1], counts[i]))
        self.project = torch.nn.Conv1d(32, 1, 1, bias=False)

    def embed(self, time):
        time = time * self.positions
        return torch.stack((torch.sin(time), torch.cos(time)), 2).flatten(1, 2)
    
    def forward(self, x, t):
        t = self.embed(t).expand(-1, -1, x.shape[2])
        n = self.embed(x[:, -2:-1, :])
        e = self.embed(x[:, -3:-2, :])
        x = torch.cat((x[:, :-3], x[:, -1:], t), 1)
        buffers = [self.downs[0](x) + torch.cat((n, e), 1)]
        for layer in self.downs[1:]:
            buffers.append(layer(buffers[-1]))
        output = self.ups[0](buffers[-1])
        for layer, skip in zip(self.ups[1:], reversed(buffers)):
            output = layer(torch.cat((output, skip), 1))
        return self.project(output)
    
    @classmethod
    def load(cls, path):
        dummy_data = torch.rand(1, 7, 1024)
        dummy_time = torch.ones(1, 1, 1)
        model = Reflow()
        model.load_state_dict(torch.load(path))
        model.eval()
        model = torch.jit.trace(model, (dummy_data, dummy_time))
        return model
