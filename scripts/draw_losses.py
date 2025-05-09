from pandas import DataFrame, read_csv
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np

sns.set_theme()
# plt.figure(figsize=(16, 9))
# df = read_csv(".temp/tuning.csv")

# axes = sns.lineplot(x=df["noise_std"], y=df["r"].rolling(7).apply(np.mean))
# axes.get_figure().savefig(".temp/tuning.png")

plt.figure(figsize=(16, 9))
df = read_csv(".temp/train_losses.csv")
diffdf = df.diff()
axes = sns.lineplot(
    data=df, x="epoch", y="trainloss", label="Потери на обучающем множестве", marker="o"
)
sns.lineplot(
    data=df,
    x="epoch",
    y="testloss",
    ax=axes,
    label="Потери на валидационном множестве",
    marker="o",
)
sns.lineplot(
    data=df,
    x="epoch",
    y="testauc",
    ax=axes,
    label="AUC на валидационном множестве",
    marker="o",
)
# sns.lineplot(
#     x=df["epoch"], y=diffdf["testloss"], ax=axes, label="Test Loss Delta", marker="o"
# )
# sns.lineplot(data=df, x="epoch", y="lr", ax=axes, label="Learning Rate", marker="o")
# axes.axhline(df["testauc"].max(), c="black", linestyle="--")
# axes.axvline(df["testauc"].argmax(), c="green", linestyle="--")
axes.add_patch(
    Rectangle(
        (
            df["testauc"].argmax() - 0.2,
            df["testauc"].max() - 0.01,
        ),
        0.4,
        0.02,
        edgecolor="green",
        facecolor="none",
    )
)
axes.set_xticks(range(0, df["epoch"].max() + 1), range(1, df["epoch"].max() + 2))
axes.set_ylim(0, 1)
axes.set_ylabel("Значение")
axes.set_xlabel("Эпоха")
axes.get_figure().savefig(".temp/train_losses.png")

if "inter" in argv:
    plt.show()
