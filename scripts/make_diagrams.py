from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()
plt.figure(figsize=(16, 9))

df = read_csv("dataset/metrics.csv")
df["CFG"] += 1

dfdiff = df.copy()
dfdiff["CNN-MO"] -= df["Human"]
dfdiff["CHOss"] -= df["Human"]
dfdiff["CHO"] -= df["Human"]
dfdiff["Human"] -= df["Human"]

axes = sns.lineplot(
    data=dfdiff,
    x="CFG",
    y="Human",
    # marker="o",
    label="Человек",
)

sns.lineplot(
    data=dfdiff,
    x="CFG",
    y="CNN-MO",
    marker="D",
    c="orange",
    linestyle=":",
    label="CNN-MO",
    ax=axes,
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHOss",
    yerr="CHOss_std",
    capsize=4,
    fmt="^:g",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHO",
    yerr="CHO_std",
    capsize=4,
    fmt="s:r",
)

axes.set_ylabel("ΔAUC")
axes.set_ylim(-0.4, 0.4)
# axes.set_xticks(range(1, 9), labels=dfdiff["CFG"])
axes.axvline(4.5, c="black", linestyle="-.")
axes.legend(loc=1)
axes.get_figure().savefig("dataset/fig1.png")


plt.figure(figsize=(7, 9))
bar_names = ["Human", "CNN-MO", "CHOss", "CHO"]
bars = DataFrame(
    {"Values": df[bar_names].mean().to_numpy(), "Names": ["Человек"] + bar_names[1:]}
)

axes = sns.barplot(bars, x="Names", y="Values", hue="Names")
axes.set_ylabel("Средняя AUC")
axes.set_xlabel("Наблюдатель")
axes.set_yticks(np.arange(0.5, 1.05, 0.05))
axes.set_ylim(0.5, 1)
axes.axhline(bars["Values"][0], c="black", linestyle="--")
axes.get_figure().savefig("dataset/fig2.png")
