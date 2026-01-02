 
import os
import torch
import lightning
import torchvision
import matplotlib.pyplot as pyplot

from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm.auto import tqdm

class Trainer(lightning.LightningModule):
    
    def __init__(self, model, step_count, total_step):
        super().__init__()
        self.model = model
        self.step_count = step_count
        self.total_step = total_step
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), 2e-4)
    
    def on_train_start(self):
        self.progress_bar = tqdm(
            total=self.total_step,
            desc=f"Step",
            initial=self.global_step,
            position=0,
            leave=True)
    
    def training_step(self, batch, _):
        data, target, _, mask = batch
        with torch.no_grad():
            noise = torch.randn_like(target)
            time = torch.randint(
                self.step_count, 
                size=(len(data), 1, 1), 
                device=data.device)
            alpha = time / self.step_count
            noisy = alpha * target + (1 - alpha) * noise
            noisy = torch.cat((data, noisy), 1)
            target = target - noise            
        output = self.model(noisy, time)
        loss = ((output - target).abs() * mask).sum() / mask.sum()
        self.log("loss", loss)
        self.progress_bar.n = self.global_step
        self.progress_bar.last_print_n = self.global_step
        self.progress_bar.refresh()
        return loss
    
    def on_train_end(self):
        self.progress_bar.close()
    
    def validation_step(self, batch, _):
        data, target, base, _ = batch
        result = torch.randn_like(target)
        for i in range(self.step_count):
            time = torch.ones(len(data), 1, 1, device=data.device) * i
            noisy = torch.cat((data, result), 1)
            output = self.model(noisy, time)
            result = result + output * (1 - time / self.step_count)
            alpha = (time + 1) / self.step_count
            result = result * alpha + torch.randn_like(result) * (1 - alpha)
        result = torch.sign(result) * ((result.abs() * 2).exp() - 1)
        result = result + base
        rows = int(len(result) ** 0.5)
        columns = int(len(result) / rows + 0.5)
        for i in range(len(result)):
            pyplot.subplot(rows, columns, i + 1)
            pyplot.plot(base[i, 0].cpu().detach().numpy(), linewidth=1, color="tab:orange")
            # pyplot.plot((base + target)[i, 0].cpu().detach().numpy(), linewidth=1)
            pyplot.plot(result[i, 0].cpu().detach().numpy(), linewidth=1, color="tab:blue")
            pyplot.ylim(20, 80)
        pyplot.tight_layout()
        pyplot.savefig("validation.png", dpi=150)
        pyplot.clf()
        pyplot.close()
        image = torchvision.io.read_image("validation.png")
        self.logger.experiment.add_image("validation", image, self.global_step)
        os.remove("validation.png")
