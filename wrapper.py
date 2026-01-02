
import torch

class Wrapper(torch.nn.Module):

    def __init__(self, model, step_count):
        super().__init__()
        self.model = model
        self.step_count = step_count
        self.blur = torch.nn.Conv1d(1, 1, 13, 1, 6, 1, 1, False, "replicate")
        with torch.no_grad():
            for param in self.blur.parameters():
                param.requires_grad = False
                param.data.copy_(torch.FloatTensor([[[
                    0.01219452, 0.02158901, 0.04725533, 0.08231669, 
                    0.11737847, 0.14304568, 0.15244057, 0.14304568, 
                    0.11737847, 0.08231669, 0.04725533, 0.02158901, 
                    0.01219452]]]))
                
    def forward(self, x):
        n = x.shape[0]
        base = self.blur(self.blur(x[:, 0:1, :]))
        mask1 = torch.where(base[:, 0:1, 1:] > 0, 1, 0)
        mask2 = torch.where(base[:, 0:1, :-1] > 0, 1, 0)
        deltas = (base[:, 0:1, 1:] - base[:, 0:1, :-1]) * mask1 * mask2
        deltas = torch.nn.functional.pad(deltas, (1, 0), "reflect")
        deltas = torch.concat((deltas, x[:, 1:]), 1)
        result = torch.randn_like(base)
        i = torch.tensor(0)
        while i < self.step_count:
            time = torch.ones(n, 1, 1, device=x.device) * i
            noisy = torch.cat((deltas, result), 1)
            output = self.model(noisy, time)
            result = result + output * (1 - time / self.step_count)
            alpha = (time + 1) / self.step_count
            result = result * alpha + torch.randn_like(result) * (1 - alpha)
            i += 1
        result = torch.sign(result) * ((result.abs() * 2).exp() - 1)
        return self.blur(result + base)
