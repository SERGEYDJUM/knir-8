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
dfdiff["CHOss(r)"] -= df["Human"]
dfdiff["CHO"] -= df["Human"]
dfdiff["CHO(r)"] -= df["Human"]

# dfdiff["NPWMF"] -= df["Human"]
dfdiff["Human"] -= df["Human"]

axes = sns.lineplot(
    data=dfdiff,
    x="CFG",
    y="Human",
    # marker="o",
    label="Человек",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CNN-MO",
    yerr="CNN-MO_std",
    capsize=5,
    fmt="D:",
    c="red",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHOss",
    yerr="CHOss_std",
    capsize=5,
    fmt="v:",
    c="seagreen",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHOss(r)",
    yerr="CHOss(r)_std",
    capsize=5,
    fmt="^:",
    c="mediumseagreen",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHO",
    yerr="CHO_std",
    capsize=5,
    fmt="<:",
    c="darkmagenta",
)

axes.errorbar(
    data=dfdiff,
    x="CFG",
    y="CHO(r)",
    yerr="CHO(r)_std",
    capsize=5,
    fmt=">:",
    c="magenta",
)

# axes.errorbar(
#     data=dfdiff,
#     x="CFG",
#     y="NPWMF",
#     yerr="NPWMF_std",
#     capsize=4,
#     fmt="s:m",
# )

axes.set_ylabel("Δ AUC")
axes.set_ylim(-0.4, 0.4)
axes.set_xticks(range(1, 11), labels=dfdiff["CFG"])
axes.set_xlabel("Номер конфигурации")
axes.axvline(4.5, c="black", linestyle="-.")
axes.axvline(8.5, c="black", linestyle="-.")
axes.legend(loc=1)
axes.get_figure().savefig("dataset/fig1.png")


plt.figure(figsize=(7, 9))
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

fig, axes = plt.subplots()
axes.bar(bars["Names"], bars["Values"], color=bar_colors)
axes.set_ylabel("Средняя AUC")
axes.set_xlabel("Наблюдатель")
axes.set_yticks(np.arange(0.5, 1.05, 0.05))
axes.set_ylim(0.5, 1)
axes.axhline(bars["Values"][0], c="black", linestyle="--")
axes.get_figure().savefig("dataset/fig2.png")
