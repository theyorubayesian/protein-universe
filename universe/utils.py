"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""
import torch


def save_checkpoint(epoch: int, epoch_step, model, optimizer, path: str, valid_loss: float = None):
    checkpoint = {
        'epoch': epoch + 1,
        'epoch_step': epoch_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'valid_loss': valid_loss
    }
    torch.save(checkpoint, path)
