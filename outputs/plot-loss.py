import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ema(x, alpha=0.1):
    """Exponential moving average with smoothing factor alpha in (0,1]."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

df = pd.read_csv('loss.csv')
expected = {"Wall time", "Step", "Value"}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

df = df.copy()
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Step", "Value"]).sort_values("Step").reset_index(drop=True)

steps = df["Step"].to_numpy()
values = df["Value"].to_numpy()

figsize = (8.0, 4.5)   # inches
dpi = 200              # 8*200 x 4.5*200 = 1600 x 900 px

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.plot(steps, values, color='royalblue', alpha=0.35, label='Raw')
ax.plot(steps, ema(values, alpha=0.1), color='royalblue', alpha=1, label='EMA')

xticks = np.arange(0, 1_000_001, 200_000)   # 0, 200K, ..., 1M
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x//1000}K" if x < 1_000_000 else "1M" for x in xticks])

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Pre-training: MLM + NSP Loss")
ax.set_ylim((3.5, 11.5))
ax.legend()
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

fig.tight_layout()
fig.savefig("loss.png", dpi=300, bbox_inches="tight")
plt.show()