
import torch
from blocks import Conv, Down, Up, Residual

class Reflow(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        counts = [64, 80, 96, 112, 128, 144, 160]
        positions = torch.pow(10000, -torch.arange(8) / 8)[None, :, None]
        self.register_buffer("positions", positions)
        self.downs = torch.nn.ModuleList([torch.nn.Conv1d(5 + 16 + 32, counts[0], 3, 1, 1)])
        self.expand = torch.nn.Conv1d(32, 64, 1)
        for i in range(len(counts) - 1):
            self.downs.append(Down(counts[i], counts[i + 1], int(128 / (2 ** i))))
        self.ups = torch.nn.ModuleList([Residual(counts[-1], 2)])
        for i in reversed(range(len(counts) - 1)):
            self.ups.append(Up(2 * counts[i + 1], counts[i], int(128 / (2 ** i))))
        self.project = torch.nn.Conv1d(counts[0], 1, 1, bias=False)

    def embed(self, time):
        time = time * self.positions
        return torch.stack((torch.sin(time), torch.cos(time)), 2).flatten(1, 2)
    
    def forward(self, x, z, t):
        t = self.embed(t).expand(-1, -1, x.shape[2])
        n = self.embed(x[:, -2:-1, :])
        e = self.embed(x[:, -3:-2, :])
        x = torch.cat((x[:, :-3], x[:, -1:], t, z), 1)
        buffers = [self.downs[0](x) + self.expand(torch.cat((n, e), 1))]
        for layer in self.downs[1:]:
            buffers.append(layer(buffers[-1]))
        output = self.ups[0](buffers[-1])
        for layer, skip in zip(self.ups[1:], reversed(buffers)):
            output = layer(torch.cat((output, skip), 1))
        return self.project(output)
    
    @classmethod
    def load(cls, path):
        dummy_data = torch.rand(1, 7, 1024)
        dummy_code = torch.rand(1, 32, 1024)
        dummy_time = torch.ones(1, 1, 1)
        model = Reflow()
        model.load_state_dict(torch.load(path))
        model.eval()
        model = torch.jit.trace(model, (dummy_data, dummy_code, dummy_time))
        return model
    
class Encoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        positions = torch.pow(10000, -torch.arange(8) / 8)[None, :, None]
        self.register_buffer("positions", positions)
        self.expand = torch.nn.Conv1d(7 + 16, 32, 3, 1, 1)
        counts = [32, 48, 64, 80, 96, 112, 128]
        self.layers = torch.nn.Sequential()
        for i in range(len(counts) - 1):
            self.layers.append(Down(counts[i], counts[i + 1], int(128 / (2 ** i))))
        for i in reversed(range(len(counts) - 1)):
            self.layers.append(Up(counts[i + 1], counts[i], int(128 / (2 ** i))))

    def forward(self, x0, x1, x, t): 
        t = self.embed(t).expand(-1, -1, x.shape[2])
        n = self.embed(x[:, -2:-1, :])
        e = self.embed(x[:, -3:-2, :])
        x = torch.cat((x[:, :-3], x[:, -1:], x0, x1, t), 1)
        return self.layers(self.expand(x) + torch.cat((n, e), 1))
        
    def embed(self, time):
        time = time * self.positions
        return torch.stack((torch.sin(time), torch.cos(time)), 2).flatten(1, 2)

class Critic(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(64, 128, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(128, 256, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, 256, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, 256, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, 256, 7, 2, 3), torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, 1, 16))
        
    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    import torchinfo
    model = Reflow()
    encoder = Encoder()
    critic = Critic()
    dummy = torch.rand(1, 7, 1024)
    noise = torch.rand(1, 1, 1024)
    target = torch.rand(1, 1, 1024)
    time = torch.ones(1, 1, 1)
    code = encoder(noise, target, dummy, time)
    torchinfo.summary(encoder, input_data=(noise, target, dummy, time))
    torchinfo.summary(model, input_data=(dummy, code, time))
    torchinfo.summary(critic, input_data=code)
    torch.onnx.export(encoder, (noise, target, dummy, time), "e.onnx", dynamo=False)
    torch.onnx.export(model, (dummy, code, time), "t.onnx", dynamo=False)
    torch.onnx.export(critic, code, "c.onnx", dynamo=False)
    import subprocess, os
    subprocess.run(["onnxsim", "t.onnx", "test.onnx"])
    os.remove("t.onnx")
    subprocess.run(["onnxsim", "e.onnx", "test_encoder.onnx"])
    os.remove("e.onnx")
    subprocess.run(["onnxsim", "c.onnx", "test_critic.onnx"])
    os.remove("c.onnx")
    