```markdown
# DD2360 GPU Programming — HW2 (CUDA Optimization)

Solution repository for **DD2360 GPU Programming (KTH)** — **Homework 2** (Group 18).  
This homework focuses on CUDA performance techniques including:

- **Shared memory + atomics** (Histogram with saturation)
- **Parallel reduction** (CPU vs GPU timing comparison)
- **Tiled matrix multiplication (GEMM)** + experiment plots (Q5)

---

## Repository Layout

> Your repo may either contain the folder `DD2360HT25_HW2_Group18/` at the top level, or its contents directly.  
> The structure below reflects the original zip structure.

```

DD2360HT25_HW2_Group18/
├── DD2360HT25_HW2_Group18.pdf        # Report (optional to keep public)
├── data_Lab2-main/                   # Extra notes/data files (not required to run code)
├── Q1/                               # Histogram (shared memory + atomics)
│   ├── q1.cu
│   ├── Makefile
│   ├── q1.md                         # Written answers
│   └── plot/
│       ├── plot.py
│       ├── hist_uniform_*.txt
│       ├── hist_normal_*.txt
│       ├── uniform.png
│       └── normal.png
├── Q2/                               # Reduction (CPU vs GPU)
│   ├── q2.cu
│   ├── Makefile
│   ├── q2.sh                         # Batch benchmark runner
│   ├── q2_times.txt
│   ├── plot.py
│   ├── reduction_times.png
│   └── q2.md                         # Written answers
└── Q[3]/                              # Tiled GEMM + Q5 plots
├── matmul.cu
├── makefile.txt                  # use: make -f makefile.txt
├── plot_q5_matmul.py
├── q5_output.txt
├── q5_matmul_bars.png
├── q5_matmul_gpu_bars.png
└── reademe.txt                   # Notes/instructions for Q3

````

---

## Requirements

- **NVIDIA GPU** + CUDA support
- **CUDA Toolkit** (`nvcc`)
- OS: Linux / WSL / Windows (CUDA configured)
- (Optional, for plots) **Python 3** + `matplotlib` + `numpy`

---

## CUDA Architecture (`-arch=sm_XX`)

`Q1/Makefile` and `Q2/Makefile` compile with `-arch=sm_89` by default.  
If your GPU has a different compute capability, edit `NVCC_FLAGS` in the Makefile, e.g.:

- `sm_86` (RTX 30xx)
- `sm_89` (RTX 40xx laptop variants)
- `sm_80` (A100)

---

## Q1 — Histogram (Shared Memory + Atomics)

### Build
```bash
cd Q1
make
````

### Run

```bash
./q1
```

`q1` generates histogram text files in the **current directory**:

* `hist_uniform_1024.txt`, `hist_uniform_10240.txt`, ...
* `hist_normal_1024.txt`, `hist_normal_10240.txt`, ...

### Plot (bar charts)

The plotting script in `Q1/plot/plot.py` expects the `hist_*.txt` files in the **same directory** where you run it.

Option A (recommended):

```bash
# from Q1/
mv hist_*.txt plot/
cd plot
python plot.py
```

Outputs:

* `uniform.png`
* `normal.png`

> Written answers for Q1 are in `Q1/q1.md`.

---

## Q2 — Reduction (CPU vs GPU)

### Build

```bash
cd Q2
make
```

### Run a single case

`q2` requires an argument `N`:

```bash
./q2 262144
```

### Run benchmarks (512 → 262144, ×2)

```bash
bash q2.sh
```

This writes results to:

* `q2_times.txt`

### Plot CPU vs GPU time

```bash
python plot.py
```

Output:

* `reduction_times.png`

> Written answers for Q2 are in `Q2/q2.md`.

---

## Q[3] — Tiled Matrix Multiplication (GEMM) + Q5 Experiments

This part contains a CUDA implementation for:

* Naive GEMM kernel
* Multiple tiled kernels (different tile sizes)
* Q5 experiment output parsing + bar plots

### Build

```bash
cd "Q[3]"
make -f makefile.txt
```

Optional: specify architecture:

```bash
make -f makefile.txt CUDA_ARCH=sm_89
```

### Run once (prints timings)

```bash
make -f makefile.txt run
# or directly:
./matmul
```

### Run Q5 experiment (save log) + generate plots

```bash
make -f makefile.txt q5
make -f makefile.txt plot
```

Outputs:

* `q5_output.txt`
* `q5_matmul_bars.png` (CPU + all GPU kernels)
* `q5_matmul_gpu_bars.png` (GPU kernels only)

Clean:

```bash
make -f makefile.txt clean
```

---

## Results (Figures Included)

* Q1: `Q1/plot/uniform.png`, `Q1/plot/normal.png`
* Q2: `Q2/reduction_times.png`
* Q3: `Q[3]/q5_matmul_bars.png`, `Q[3]/q5_matmul_gpu_bars.png`

---

## Notes on Public Repositories

For a cleaner public repo, consider **removing compiled binaries** (build outputs), e.g.:

* `Q1/q1`, `Q2/q2`, `Q[3]/*.exe`

Also consider whether to keep course/internal files (e.g., some items under `data_Lab2-main/`) public.

---

## Academic Integrity

This repository is published for **learning/portfolio** purposes.
If you are currently taking this course, please follow your institution’s **academic integrity policy**.

---

## Suggested GitHub Topics

Add these under **About → Topics** for better search:
`cuda`, `gpu-programming`, `dd2360`, `kth`, `shared-memory`, `atomics`, `reduction`, `tiled-matrix-multiplication`, `benchmark`

---

## License

MIT License.