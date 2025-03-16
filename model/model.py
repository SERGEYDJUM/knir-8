from torch.optim.lr_scheduler import StepLR
from torch.optim import Adadelta
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch


from .data_load import MyDataset


class CNNMO(nn.Module):
    def __init__(self):
        super(CNNMO, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3)
        self.conv_2 = nn.Conv2d(16, 8, 3)
        self.dropout_1 = nn.Dropout(0.2)
        self.lin_1 = nn.Linear(8 * 30**2, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.lin_2 = nn.Linear(128, 1)

    def forward(self, x):
        # Convolutions
        x = F.relu(self.conv_1(x))
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
        pred = model(data)
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        batches = max(batches, i)

    print(f"Epoch {epoch+1}: \n\tAverage batch loss: {loss_sum/batches:.4f}")


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            pred, target = model(data.to(device)), target.to(device)

            matched = (
                torch.logical_xor(F.sigmoid(pred) < 0.5, target.to(torch.bool))
                .to(torch.float32)
                .mean()
            )

            auc = roc_auc_score(target.flatten().cpu(), pred.flatten().cpu())

            print(f"\tTest set class separation: {matched * 100:.2f}%")
            print(f"\tTest set ROC AUC: {auc:.4f}")


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
        default=8,
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
        default=0.96,
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
        "--seed", type=int, default=1337, metavar="S", help="random seed (default: 1)"
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
        MyDataset(train=True, augments=40),
        batch_size=args.batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, augments=20),
        batch_size=args.test_batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    model = CNNMO().to(device)
    optimizer = Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

        print(f"\tLatest learning rate: {scheduler.get_last_lr()[0]:.4f}\n")

        if args.dry_run:
            break

    if args.save_model:
        torch.save(model.state_dict(), "checkpoints/model.pt")
