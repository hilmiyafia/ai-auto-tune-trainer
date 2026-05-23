 
import os
import torch
import lightning
import torchvision
import matplotlib.pyplot as pyplot

from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm.auto import tqdm

class Trainer(lightning.LightningModule):
    
    def __init__(self, model, encoder, critic, step_count, total_step):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.critic = critic
        self.step_count = step_count
        self.total_step = total_step
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        model_parameters = (
            list(self.model.parameters()) 
            + list(self.encoder.parameters()))
        return [
            torch.optim.AdamW(model_parameters, 2e-4),
            torch.optim.AdamW(self.critic.parameters(), 2e-4)]
    
    def on_train_start(self):
        self.progress_bar = tqdm(
            total=self.total_step,
            desc=f"Step",
            initial=self.global_step,
            position=0,
            leave=True)
        
    def training_step(self, batch, batch_index):
        model_opt, critic_opt = self.optimizers()
        if batch_index % 2 == 0:
            self.training_critic(batch, critic_opt)
        else:
            self.training_model(batch, model_opt)
        
    def unpack_data(self, batch):
        data, real, _, mask = batch
        with torch.no_grad():
            noise = torch.randn_like(real)
            time = torch.randint(
                self.step_count, 
                size=(len(data), 1, 1), 
                device=data.device)
            alpha = time / self.step_count
            noisy = alpha * real + (1 - alpha) * noise
            noisy = torch.cat((data, noisy), 1)
            target = real - noise   
        return real, noise, noisy, target, time, mask

    def training_critic(self, batch, opt):
        self.toggle_optimizer(opt)
        opt.zero_grad()
        real, noise, noisy, _, time, _ =  self.unpack_data(batch)
        with torch.no_grad():
            code = self.encoder(noise, real, noisy, time)
            normal = torch.randn_like(code)
        loss_normal = (1 - self.critic(normal)).square().mean()
        loss_code = (1 + self.critic(code)).square().mean()
        self.log("critic/normal", loss_normal)
        self.log("critic/code", loss_code)
        self.manual_backward(loss_normal + loss_code)
        opt.step()
        self.untoggle_optimizer(opt)
    
    def training_model(self, batch, opt):
        self.toggle_optimizer(opt)
        opt.zero_grad()
        real, noise, noisy, target, time, mask = self.unpack_data(batch)
        code = self.encoder(noise, real, noisy, time)      
        output = self.model(noisy, code, time)
        loss = ((output - target).abs() * mask).sum() / mask.sum()
        loss_code = (1 - self.critic(code)).square().mean()
        self.log("model/loss", loss)
        self.log("model/code", loss_code)
        self.progress_bar.n = self.global_step
        self.progress_bar.last_print_n = self.global_step
        self.progress_bar.refresh()
        self.manual_backward(loss + loss_code)
        opt.step()
        self.untoggle_optimizer(opt)
    
    def on_train_end(self):
        self.progress_bar.close()
    
    def validation_step(self, batch, _):
        data, target, base, _ = batch
        result = torch.randn_like(target)
        code = torch.randn(data.shape[0], 32, 1024).to(result.device)
        for i in range(self.step_count):
            time = torch.ones(len(data), 1, 1, device=data.device) * i
            noisy = torch.cat((data, result), 1)
            output = self.model(noisy, code, time)
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
