import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("q2_times.txt", skiprows=1)

N   = data[:, 0]
cpu = data[:, 1]
gpu = data[:, 2]

x = np.arange(len(N))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width/2, cpu, width, label="CPU")
ax.bar(x + width/2, gpu, width, label="GPU")

ax.set_xticks(x)
ax.set_xticklabels([str(int(n)) for n in N], rotation=45)
ax.set_xlabel("Array length N")
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("Reduction: CPU vs GPU time (log scale)")
ax.set_yscale("log")  # logscale for y-axis
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("reduction_times.png", dpi=300, bbox_inches="tight")
