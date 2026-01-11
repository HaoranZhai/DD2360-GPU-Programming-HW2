import os
import re
import matplotlib.pyplot as plt

# 1) 找到输出文件（默认叫 q5_output.txt）
log_path = None
candidates = ["q5_output.txt", "matmul_q5_output.txt", "matmul_output.txt"]

for name in candidates:
    if os.path.exists(name):
        log_path = name
        break

if log_path is None:
    # 尝试自动搜一搜
    for fn in os.listdir("."):
        low = fn.lower()
        if low.startswith("q5") and low.endswith(".txt"):
            log_path = fn
            break

if log_path is None:
    raise FileNotFoundError("找不到 q5 输出文件，请先运行: matmul_q5.exe > q5_output.txt")

print("Using log file:", log_path)

text = open(log_path, "r", encoding="utf-8", errors="ignore").read()

# 2) 用正则解析每个 case 的 n 和时间
pat_case  = re.compile(r"\[Q5\]\s*Running case with n\s*=\s*(\d+)", re.I)
pat_time  = re.compile(r"time\s*=\s*([0-9.]+)\s*ms", re.I)
pat_tiled = re.compile(
    r"Tiled kernel\s*\(\s*(\d+)\s*x\s*(\d+)\s*\).*time\s*=\s*([0-9.]+)\s*ms",
    re.I
)

sizes  = []
cpu    = []
naive  = []
t8     = []
t16    = []
t32    = []

cur_n = None
cur_cpu = None
cur_naive = None
cur_tiles = {}

for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
        continue

    # 新的一组 n
    m_case = pat_case.search(line)
    if m_case:
        # 先把上一组写入数组
        if cur_n is not None:
            sizes.append(cur_n)
            cpu.append(cur_cpu)
            naive.append(cur_naive)
            t8.append(cur_tiles.get(8))
            t16.append(cur_tiles.get(16))
            t32.append(cur_tiles.get(32))

        cur_n = int(m_case.group(1))
        cur_cpu = None
        cur_naive = None
        cur_tiles = {}
        continue

    # CPU 时间
    if "CPU done" in line:
        m = pat_time.search(line)
        if m:
            cur_cpu = float(m.group(1))
        continue

    # naive kernel 时间
    if line.startswith("Naive kernel done"):
        m = pat_time.search(line)
        if m:
            cur_naive = float(m.group(1))
        continue

    # tiled kernel 时间（8x8 / 16x16 / 32x32）
    if line.startswith("Tiled kernel"):
        m = pat_tiled.search(line)
        if m:
            T = int(m.group(1))          # tile 大小 8 或 16 或 32
            t_ms = float(m.group(3))     # 对应时间
            cur_tiles[T] = t_ms
        continue

# 处理最后一组
if cur_n is not None:
    sizes.append(cur_n)
    cpu.append(cur_cpu)
    naive.append(cur_naive)
    t8.append(cur_tiles.get(8))
    t16.append(cur_tiles.get(16))
    t32.append(cur_tiles.get(32))

if not sizes:
    raise RuntimeError("没有解析到任何 [Q5] case，请检查 q5_output.txt 格式。")

print("Parsed sizes:", sizes)
print("CPU times:", cpu)
print("Naive times:", naive)
print("Tiled 8x8 times:", t8)
print("Tiled 16x16 times:", t16)
print("Tiled 32x32 times:", t32)

# 3) 画并排柱状图（CPU + naive + 三个 tiled）
# 3) 图一：CPU + Naive + 三个 tiled（保留）
x = list(range(len(sizes)))
width = 0.15  # 每个小柱之间的偏移

plt.figure(figsize=(10, 6))

plt.bar([i - 2*width for i in x], cpu,   width, label="CPU")
plt.bar([i -     width for i in x], naive, width, label="Naive")
plt.bar(x,                      t8,    width, label="Tiled 8x8")
plt.bar([i +     width for i in x], t16,  width, label="Tiled 16x16")
plt.bar([i + 2*width for i in x], t32,  width, label="Tiled 32x32")

plt.xticks(x, [str(n) for n in sizes], rotation=45, ha="right")
plt.xlabel("Matrix size n (n x n)")
plt.ylabel("Runtime (ms)")
plt.title("Matrix Multiplication – Runtime vs Matrix Size (CPU + GPU)")
plt.legend()
plt.tight_layout()

out_png = "q5_matmul_bars.png"
plt.savefig(out_png, dpi=200)
print("Saved figure to", out_png)

# 4) 图二：只看 GPU（去掉 CPU）
plt.figure(figsize=(10, 6))

# 这次不画 cpu，只画 Naive + 三个 tiled
width_gpu = 0.2
plt.bar([i - 1.5*width_gpu for i in x], naive,  width_gpu, label="Naive")
plt.bar([i - 0.5*width_gpu for i in x], t8,     width_gpu, label="Tiled 8x8")
plt.bar([i + 0.5*width_gpu for i in x], t16,    width_gpu, label="Tiled 16x16")
plt.bar([i + 1.5*width_gpu for i in x], t32,    width_gpu, label="Tiled 32x32")

plt.xticks(x, [str(n) for n in sizes], rotation=45, ha="right")
plt.xlabel("Matrix size n (n x n)")
plt.ylabel("Runtime (ms)")
plt.title("Matrix Multiplication – Runtime vs Matrix Size (GPU only)")
plt.legend()
plt.tight_layout()

out_png_gpu = "q5_matmul_gpu_bars.png"
plt.savefig(out_png_gpu, dpi=200)
print("Saved figure to", out_png_gpu)

try:
    plt.show()
except Exception as e:
    print("GUI show failed, but PNGs are saved:", e)

