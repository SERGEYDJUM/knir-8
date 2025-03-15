from torch.optim.lr_scheduler import StepLR
from torch.optim import Adadelta
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch


from .data_load import MyDataset


class CNNMO(nn.Module):
    def __init__(self):
        super(CNNMO, self).__init__()

        self.conv_1 = nn.Conv2d(1, 8, 3)
        self.conv_2 = nn.Conv2d(8, 16, 3)

        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)
        self.lin_1 = nn.Linear(16 * (14 * 14), 128)
        self.lin_2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutions
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)

        # Perceptron
        x = torch.flatten(x, 1)
        x = F.relu(self.lin_1(self.dropout_1(x)))
        x = self.lin_2(self.dropout_2(x))

        # Output logits
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_sum = 0
    batches = 0

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(data), target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        batches = max(batches, i)

    print(f"Epoch {epoch}: \n\tAverage batch loss: {loss_sum/batches:.6f}")


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            pred = F.sigmoid(model(data.to(device)).cpu())
            matched = ((pred > 0.5) == (target > 0.5)).sum() / target.shape[0]

            print(f"\tTest set correct guesses: {matched * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10000,
        metavar="N",
        help="input batch size for testing (default: 10000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu")
    cuda_enabled = False

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_enabled = True

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, augments=30),
        batch_size=args.batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, augments=10),
        batch_size=args.test_batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    model = CNNMO().to(device)
    optimizer = Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        print(f"\tLatest learning rate: {scheduler.get_last_lr()[0]:.4f}\n")

    if args.save_model:
        torch.save(model.state_dict(), "checkpoints/model.pt")
