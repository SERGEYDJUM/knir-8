from torch.optim.lr_scheduler import StepLR
from torch.optim import Adadelta
from sklearn.metrics import roc_auc_score
from PIL import Image
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch


from .data_load import MyDataset

_precision_record = 0


class CNNMO(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        conv_1_chs: int = 16,
        conv_2_chs: int = 16,
        res_chs: int = 32,
        lin_2_neurons: int = 64,
    ) -> None:
        super(CNNMO, self).__init__()

        # ResNet Block
        self.res_conv_1 = nn.Conv2d(
            1, res_chs, 3, bias=False, padding=1, padding_mode="replicate"
        )
        self.res_bn = nn.BatchNorm2d(res_chs)
        self.res_conv_2 = nn.Conv2d(res_chs, 1, 1, bias=False)

        # Convolutional Block
        self.conv_1 = nn.Conv2d(1, conv_1_chs, 3)
        self.conv_2 = nn.Conv2d(conv_1_chs, conv_2_chs, 3)

        # Perceptron
        self.dropout_1 = nn.Dropout(0.1)
        self.lin_1 = nn.Linear(conv_2_chs * ((input_size // 2 - 2) ** 2), lin_2_neurons)
        self.dropout_2 = nn.Dropout(0.5)
        self.lin_2 = nn.Linear(lin_2_neurons, 1)

    def res_block_forward(self, x: Tensor) -> Tensor:
        r = self.res_conv_1(x)
        r = F.elu(self.res_bn(r))
        r = self.res_conv_2(r)
        return F.elu(r + x)

    def convs_forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        return torch.flatten(x, 1)

    def perceptron_forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.lin_1(self.dropout_1(x)))
        x = self.lin_2(self.dropout_2(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.res_block_forward(x)
        x = self.convs_forward(x)
        x = self.perceptron_forward(x)
        return x


class RNMO(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        res_blocks: int = 5,
        lin_2_neurons: int = 64,
    ) -> None:
        super(RNMO, self).__init__()

        # ResNet Blocks
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            max(2 * bidx, 1),
                            2 * (bidx + 1),
                            3,
                            bias=False,
                            padding=1,
                            padding_mode="replicate",
                        ),
                        nn.BatchNorm2d(2 * (bidx + 1)),
                        nn.Conv2d(
                            2 * (bidx + 1),
                            2 * (bidx + 1),
                            3,
                            bias=False,
                            padding=1,
                            padding_mode="replicate",
                        ),
                    ],
                )
                for bidx in range(res_blocks)
            ]
        )

        # Perceptron
        self.dropout_1 = nn.Dropout(0.25)
        self.lin_1 = nn.Linear(input_size * input_size, lin_2_neurons)
        self.dropout_2 = nn.Dropout(0.5)
        self.lin_2 = nn.Linear(lin_2_neurons, 1)

    def res_block_forward(self, bidx: int, x: Tensor) -> Tensor:
        r = self.blocks[bidx][0](x)
        r = F.elu(self.blocks[bidx][1](r))
        r = self.blocks[bidx][2](r)
        return F.elu(r + x)

    def perceptron_forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.lin_1(self.dropout_1(x)))
        x = self.lin_2(self.dropout_2(x))
        return x

    def forward(self, x: Tensor) -> Tensor:
        for bidx in range(len(self.blocks)):
            x = self.res_block_forward(bidx, x)
        x = x.flatten(1)
        x = self.perceptron_forward(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    loss_sum = 0
    batches = 0

    for i, (data, target, _) in enumerate(train_loader):
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
    global _precision_record
    model.eval()

    with torch.no_grad():
        for data, target, gt in test_loader:
            pred, target = model(data.to(device)), target.to(device)

            matched = (
                torch.logical_xor(F.sigmoid(pred) < 0.5, target.to(torch.bool))
                .to(torch.float32)
                .mean()
            )

            print(f"\tTest set precision: {matched * 100:.2f}%")

            pred, target = pred.flatten().cpu(), target.cpu()

            auc = roc_auc_score(gt.cpu(), pred)
            print(f"\tTest ROC AUC as Model Observer: {auc:.4f}")

            hauc = roc_auc_score(target, pred)
            print(f"\tTest ROC AUC as classifier: {hauc:.4f}")

            _precision_record = max(hauc, _precision_record)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument(
        "--rn",
        action="store_true",
        default=False,
        help="train a ResNet",
    )
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
        default=32,
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
        default=0.99,
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
    global _precision_record
    args = parse_args()

    device = torch.device("cpu")
    cuda_enabled = False

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_enabled = True

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, random_state=args.seed, train_split=0.9),
        batch_size=args.batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, random_state=args.seed, train_split=0.9),
        batch_size=args.test_batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    model = (RNMO() if args.rn else CNNMO()).to(device)
    print("Model weight count:", sum(p.numel() for p in model.parameters()))

    optimizer = Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        # test(model, device, test_loader)
        scheduler.step()

        print(f"\tLatest learning rate: {scheduler.get_last_lr()[0]:.4f}\n")

        if args.dry_run:
            break

    print(f"Best classifier test AUC: {_precision_record:.3f}")

    if args.save_model:
        torch.save(
            model.state_dict(), f"checkpoints/{'rn' if args.rn else 'cnn'}_mo.pt"
        )

    # def save_img(tensor: torch.Tensor, path: str):
    #     t = tensor.clone()
    #     t -= t.min()
    #     t *= 255 / t.max()
    #     Image.fromarray(t[0, 0, :, :].to(dtype=torch.uint8, device="cpu").numpy()).save(
    #         path
    #     )

    # testimg = test_loader.dataset[0][0][torch.newaxis, :, :, :].cpu()
    # save_img(testimg, ".temp/test_input.png")

    # resres = model.res_forward(testimg.to("cuda")).cpu()
    # save_img(resres, ".temp/test_resres.png")

    # save_img(resres - testimg, ".temp/test_diff.png")

    if args.save_model:
        torch.onnx.export(
            model,
            test_loader.dataset[0][0][torch.newaxis, :, :, :].to(device="cuda"),
            ".temp/model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
