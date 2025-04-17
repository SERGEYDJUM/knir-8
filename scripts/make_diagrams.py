from pandas import read_csv
import seaborn as sns

sns.set_theme()
sns.set(rc={"figure.figsize": (16, 9)})

df = read_csv("dataset/metrics.csv")
df["CFG"] += 1

axes = sns.lineplot(
    data=df,
    x="CFG",
    y="Human",
    c="grey",
    label="Human",
)

sns.lineplot(
    data=df,
    x="CFG",
    y="CNN-MO",
    marker="D",
    c="green",
    linestyle=":",
    label="CNN-MO",
    ax=axes,
)

axes.errorbar(
    data=df,
    x="CFG",
    y="CHOss",
    yerr="CHOss_std",
    capsize=2,
    fmt="^:",
)

axes.errorbar(
    data=df,
    x="CFG",
    y="CHO",
    yerr="CHO_std",
    capsize=2,
    fmt="s:y",
)

axes.set_ylabel("AUC")
axes.set_ylim(0.4, 1.1)
axes.axvline(4.5, c="black")
axes.legend(loc=4)
axes.get_figure().savefig("dataset/fig1.png")
