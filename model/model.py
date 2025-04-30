from torch.optim.lr_scheduler import StepLR
from torch.optim import Adadelta
from sklearn.metrics import roc_auc_score

# from PIL import Image
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch


from .data_load import MyDataset


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


def train(model, device, train_loader, optimizer, epoch) -> float:
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

    avg_loss = loss_sum / batches
    print(f"Epoch {epoch+1}: \n\tAverage batch loss: {avg_loss:.4f}")
    return avg_loss


def test(model, device, test_loader):
    model.eval()

    hauc = 0
    test_loss = 0

    with torch.no_grad():
        for data, target, gt in test_loader:
            pred, target = model(data.to(device)), target.to(device)

            matched = (
                torch.logical_xor(F.sigmoid(pred) < 0.5, target.to(torch.bool))
                .to(torch.float32)
                .mean()
            )

            test_loss += F.binary_cross_entropy_with_logits(pred, target)

            print(f"\tTest set loss: {test_loss:.5f}")

            print(f"\tTest set precision: {matched * 100:.2f}%")

            pred, target = pred.flatten().cpu(), target.cpu()

            auc = roc_auc_score(gt.cpu(), pred)
            print(f"\tTest ROC AUC as Model Observer: {auc:.4f}")

            hauc = roc_auc_score(target, pred)
            print(f"\tTest ROC AUC as classifier: {hauc:.4f}")

    return test_loss, hauc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument(
        "--rn",
        action="store_true",
        default=False,
        help="train a ResNet instead (WIP)",
    )
    return parser.parse_args()


def save_model(
    model, path: str, export: bool = False, onnx_input: torch.Tensor | None = None
) -> None:
    torch.save(model.state_dict(), path)

    if export:
        torch.onnx.export(
            model,
            onnx_input,
            path + ".onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


def main():
    args = parse_args()

    device = torch.device("cpu")
    cuda_enabled = False

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_enabled = True

    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, random_state=args.seed, train_split=0.85),
        batch_size=args.batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, random_state=args.seed, train_split=0.85),
        batch_size=args.test_batch_size,
        pin_memory=cuda_enabled,
        shuffle=True,
    )

    model = (RNMO() if args.rn else CNNMO()).to(device)
    print("Model weight count:", sum(p.numel() for p in model.parameters()))

    optimizer = Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    model_basename = f"{'rn' if args.rn else 'cnn'}_mo"

    with open(".temp/train_losses.csv", "w") as trainlog:
        print("epoch,lr,trainloss,testloss,testauc", file=trainlog)

        best_auc = 0

        for epoch in range(args.epochs):
            trainloss = train(model, device, train_loader, optimizer, epoch)
            testloss, testauc = test(model, device, test_loader)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            print(f"{epoch},{lr},{trainloss},{testloss},{testauc}", file=trainlog)
            print(f"\tLatest learning rate: {lr:.4f}\n")

            if args.dry_run:
                break

            if testauc >= best_auc:
                best_auc = testauc
                if args.save_model:
                    save_model(model, f"checkpoints/{model_basename}_best.pt")

        print(f"Best classifier test AUC: {best_auc:.3f}")

    if args.save_model:
        save_model(
            model,
            f"checkpoints/{model_basename}.pt",
            export=True,
            onnx_input=test_loader.dataset[0][0][torch.newaxis, :, :, :].to(
                device="cuda"
            ),
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
