"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""
import os
import torch


def save_checkpoint(epoch: int, epoch_step, model, optimizer, path: str, valid_loss: float = None):
    checkpoint = {
        'epoch': epoch + 1,
        'epoch_step': epoch_step,
        'optimizer': optimizer.state_dict(),
        'valid_loss': valid_loss
    }
    torch.save(model, os.path.join(path, "model.pth"))
    torch.save(checkpoint, os.path.join(path, "state.pt"))
