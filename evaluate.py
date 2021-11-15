import argparse

import torch

from universe.dataset import create_dataloader


def evaluate_model(model, test_loader, device):
    correct = 0
    y_pred = []
    y_true = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            seqs, labels, _ = tuple(t.to(device) for t in batch[:2])

            output = model(seqs)
            prob = torch.nn.Softmax(output)
            arg_maxs = torch.argmax(prob, dim=1)

            correct += torch.sum(labels == arg_maxs).item()
            y_pred.extend(arg_maxs.tolist())
            y_true.extend(labels.tolist())

        accuracy = correct / len(test_loader)

    return accuracy, y_pred, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--test_data")
    parser.add_argument("--overwrite_data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    model = torch.load(args.model_path)
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = create_dataloader(
        args.test_data,
        args.overwrite_data_cache, 
        args.num_classes, 
        "val",
        args.batch_size,
        shuffle=False
    )

    accuracy, y_pred, y_true = evaluate_model(model, dataloader, device)
    print(f"Accuracy: {accuracy}")
    # TODO: Plot Confusion Matrix



