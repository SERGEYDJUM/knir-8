from pandas import DataFrame, read_csv
from sys import argv

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = read_csv("dataset/metrics.csv")
df["CFG"] += 1

dfdiff = df.copy()
dfdiff["CNN-MO"] -= df["Human"]
dfdiff["CHOss"] -= df["Human"]
dfdiff["CHOss(r)"] -= df["Human"]
dfdiff["CHO"] -= df["Human"]
dfdiff["CHO(r)"] -= df["Human"]
dfdiff["Human"] -= df["Human"]

bar_names = ["Human", "CNN-MO", "CHOss", "CHOss(r)", "CHO", "CHO(r)"]
bar_colors = [
    "steelblue",
    "red",
    "seagreen",
    "mediumseagreen",
    "darkmagenta",
    "magenta",
]
bars = DataFrame(
    {
        "Values": df.head(8)[bar_names].mean().to_numpy(),
        "Names": ["Человек"] + bar_names[1:],
    }
)

sns.set_theme()
fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[5, 3])
fig.set_figwidth(21)
fig.set_figheight(9)

sns.lineplot(
    data=dfdiff,
    x="CFG",
    y="Human",
    label="Человек",
    ax=ax1,
)

ax1.errorbar(
    data=dfdiff,
    x="CFG",
    y="CNN-MO",
    yerr="CNN-MO_std",
    capsize=5,
    fmt="D:",
    c="red",
)

ax1.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHOss",
    yerr="CHOss_std",
    capsize=5,
    fmt="v:",
    c="seagreen",
)

ax1.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHOss(r)",
    yerr="CHOss(r)_std",
    capsize=5,
    fmt="^:",
    c="mediumseagreen",
)

ax1.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHO",
    yerr="CHO_std",
    capsize=5,
    fmt="<:",
    c="darkmagenta",
)

ax1.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHO(r)",
    yerr="CHO(r)_std",
    capsize=5,
    fmt=">:",
    c="magenta",
)

ax1.set_ylabel("Δ AUC")
ax1.set_ylim(-0.4, 0.4)
ax1.set_xticks(range(1, 11), labels=dfdiff["CFG"])
ax1.set_xlabel("Номер конфигурации")
ax1.axvline(4.5, c="black", linestyle="--")
ax1.axvline(8.5, c="black", linestyle="--")
ax1.legend(loc=1)


ax2.bar(bars["Names"], bars["Values"], color=bar_colors)
ax2.set_ylabel("Средняя AUC")
ax2.set_xlabel("Наблюдатель")
ax2.set_yticks(np.arange(0.5, 1.05, 0.05))
ax2.set_ylim(0.5, 1)
ax2.axhline(bars["Values"][0], c="black", linestyle="--")

fig.savefig("dataset/fig.png")

if "inter" in argv:
    plt.show()
