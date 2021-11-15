"""
Written by: Akintunde 'theyorubayesian' Oladipo
14/Nov/2021
"""
import os

import torch
from torch.utils.data import DataLoader
from torch import nn

from universe.utils import save_checkpoint


def validate(model, val_dataloader, loss_fn, args):
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            if args.n_gpu > 0:
                seqs, labels = tuple(t.to(args.device) for t in batch[:2])
            
            # lengths = batch[-1]

            output = model(seqs) # , lengths)
            loss = loss_fn(output, labels)

            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(val_dataloader)
    return avg_valid_loss


def train_lstm(
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    model: nn.Module, 
    optimizer,
    scheduler,
    loss_fn,
    logging_client,
    num_epochs,
    args
):
    model.train()
    best_valid_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        epoch_step = 0
        n_sequences_epoch = 0
        total_loss_epoch = 0

        for batch in train_dataloader:
            if args.n_gpu > 0:
                seqs, labels = tuple(t.to(args.device) for t in batch[:2])
            
            # lengths = batch[-1]

            outputs = model(seqs) #, lengths)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)

            if (loss != loss).data.any():
                print("NaN detected in loss")
                exit()

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            last_loss = loss.item()
            total_loss_epoch += loss.item()
            epoch_step += 1
            global_step += 1

            optimizer.step()
            
            if global_step % args.logging_interval == 0:
                logging_client["losses/loss"].log(last_loss)
                logging_client["losses/cum_avg_loss_epoch"].log(total_loss_epoch / epoch_step)
                # TODO: Log memory usage

            # if global_step % args.checkpoint_interval == 0:
            #    output_dir = os.path.join(
            #        args.output_dir, f"checkpoint-{global_step}"
            #    )
            #    save_checkpoint(epoch, epoch_step, model, optimizer, output_dir + "model.pt")

            n_sequences_epoch += seqs.size(0)
        
        # -----------------------
        # End of epoch validation
        # -----------------------
        valid_loss = validate(model, val_dataloader, loss_fn, args)
        logging_client["losses/validation_loss"].log(valid_loss)

        scheduler.step(valid_loss)
        logging_client["learning_rate"].log(optimizer.param_groups[0]["lr"])
        model.train()

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            output_dir = os.path.join(
                    args.output_dir, f"val-checkpoint-{global_step}"
                )
            os.makedirs(output_dir, exist_ok=True)
            save_checkpoint(
                epoch, epoch_step, model, optimizer, output_dir, valid_loss
            )
        
        logging_client["losses/epoch_loss"].log(total_loss_epoch/len(train_dataloader))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, 
            total_loss_epoch / len(train_dataloader),
            valid_loss
            ))
