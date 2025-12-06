
import os
import torch
import lightning
import subprocess
from model import Reflow
from trainer import Trainer
from wrapper import Wrapper
from preprocess import preprocess
from dataset import Dataset, get_dataloader
from torch.utils.data import random_split
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

STEPS = 30000
STEP_COUNT = 20
VAL_INTERVAL = 1000
BATCH_SIZE = 8
PATH = r"logs\tuner\base\checkpoints"

if __name__ == "__main__":

    print("")
    print("#########################")
    print("## 1/3 PREPROCESS DATA ##")
    print("#########################")
    print("")

    preprocess()

    print("")
    print("#####################")
    print("## 2/3 TRAIN MODEL ##")
    print("#####################")
    print("")

    dataset = Dataset()
    val_set, train_set = random_split(dataset, [4, len(dataset) - 4])
    reflow = Reflow()
    logger = TensorBoardLogger("logs", "tuner", "base")

    # Load checkpoint
    checkpoint = None
    if os.path.exists(PATH):
        files = list(os.listdir(PATH))
        files.sort()
        if len(files) > 0:
            checkpoint = f"{PATH}\\{files[-1]}"

    # Train
    trainer = lightning.Trainer(
        max_steps=STEPS, 
        logger=logger,
        check_val_every_n_epoch=None,
        val_check_interval=VAL_INTERVAL,
        enable_model_summary=False,
        log_every_n_steps=3)
    trainer.fit(
        Trainer(reflow, STEP_COUNT), 
        ckpt_path=checkpoint,
        train_dataloaders=get_dataloader(train_set, BATCH_SIZE), 
        val_dataloaders=get_dataloader(val_set, BATCH_SIZE, False, False))
    
    # Save
    torch.save(reflow.state_dict(), "model.pt")
    
    print("")
    print("######################")
    print("## 3/3 EXPORT MODEL ##")
    print("######################")
    print("")

    wrapper = Wrapper(Reflow.load("model.pt"), STEP_COUNT).eval()
    wrapper = torch.jit.script(wrapper)
    dummy = torch.randn(1, 6, 1024)
    torch.onnx.export(
        wrapper, 
        dummy, 
        "model.onnx", 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "length"},
            "output": {0: "batch", 2: "length"}})
    os.remove("model.pt")

    print("")
    print("##########")
    print("## DONE ##")
    print("##########")
    print("")