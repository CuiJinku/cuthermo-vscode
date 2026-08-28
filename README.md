# cuThermo

**cuThermo** is a Visual Studio Code extension that profiles CUDA applications and visualizes GPU memory-access heatmaps. It integrates with your CUDA toolchain and the bundled `cuThermostat.so` profiler to make GPU performance analysis easier, more intuitive, and fully interactive inside VS Code.

---

## 🚀 Features

- **Run GPU profiling directly in VS Code**  
  Invoke *cuThermo: Run on Current Target* to execute the selected CUDA binary and collect memory-access traces.

- **Automatic heatmap visualization**  
  Profiling results are parsed and displayed in a custom WebView panel, showing a live heatmap of memory sector activity.

- **Duplicate-row collapsing**  
  The heatmap applies intelligent preprocessing (same logic as your web-based tool) to compress repetitive frequency rows.

- **Environment validation**  
  Includes a command to check CUDA toolkit, `nvcc`, GPU availability, and the location of `cuThermostat.so`.

- **Configurable executable selection**  
  The extension helps you select which binary to run (e.g., `add`, `matmul`, etc.).

---

## 📦 Requirements

To use cuThermo effectively, your **remote machine (with CUDA)** should have:

- **CUDA Toolkit** (nvcc available in PATH)
- **NVIDIA GPU + driver**
- **cuThermostat.so** (either bundled or installed using the extension command)

Your **local computer** (Mac/Windows/Linux) does **not** need CUDA — it only hosts the editor UI.

Optional but recommended:

- VS Code C++ extension  
- NVIDIA Nsight Systems (for advanced manual analysis)

---

# 📥 Install cuThermo

There are **two cases** depending on whether you use:

1. **VS Code locally** (opening a folder on your machine)  
2. **VS Code Remote-SSH** (connecting to a CUDA server)

---

## 1. Installing locally (simple)

### Step 1 — Download the latest `.vsix`

Latest releases are available at:

👉 https://github.com/CuiJinku/cuthermo-vscode/releases

Download the asset named:

`cuthermo-0.0.X.vsix`

<!-- screenshot: download_vsix_from_releases -->

---

### Step 2 — Install the `.vsix` in VS Code

1. Open the **Extensions** panel.
![open the Extension panel](images/open_the_extension_panel.png)

2. Click the `⋯` (More Actions) button in the top-right.
3. Choose **Install from VSIX…**
![Install from VSIX](images/install_from_vsix.png)

4. Select your downloaded `.vsix`.
![Select from local](images/select_from_local.png)


<!-- screenshot: install_from_vsix_gui -->

---

## 2. Installing on a remote CUDA server (Remote-SSH)

When profiling CUDA kernels, **cuThermo must be installed on the CUDA machine**, not your local laptop.

### Step 1 — SSH into the remote machine

```bash
ssh your_ncsu_id@remote.cluster.edu
```

### Step 2 — Download the cuThermo `.vsix` to the remote machine

Using **wget**:

```bash
wget https://github.com/CuiJinku/cuthermo-vscode/releases/download/v0.0.X/cuthermo-0.0.X.vsix
```

![wget_vsix](images/wget_vsix.png)

Or using **curl**:

```bash
curl -OL https://github.com/CuiJinku/cuthermo-vscode/releases/download/v0.0.X/cuthermo-0.0.X.vsix
```
(Replace 0.0.X with the latest version.)


<!-- screenshot: terminal_wget_vsix -->

### Step 3 — Install the extension on the remote VS Code server

Once you've connected to the server using Remote-SSH at least once, VS Code Server is installed there.
Then run:

```bash
code --install-extension cuthermo-0.0.X.vsix
```

![command install the vsix](images/code_install.png)

To update or replace an older version:

```bash
code --uninstall-extension cuthermo
code --install-extension cuthermo-0.0.X.vsix
```
<!-- screenshot: terminal_install_vsix -->

### Step 4 — Verify installation

Inside VS Code (connected via SSH):

* Open the **Extensions** panel

* Look under **Installed (Remote)**

You should see:

![installation check](images/install_check.png)
<!-- screenshot: extension_list_remote -->

## ▶️ Usage Guide

This section walks you through a typical workflow for using **cuThermo** to profile a CUDA program.

The steps assume you are already connected to the remote CUDA machine using **VS Code Remote-SSH** and have installed the cuThermo `.vsix` file on the remote host.

---

### 1. Open your CUDA project in VS Code

For demonstration, we use `gemm_naive.cu` for the tutorial.
In a terminal on the remote machine:

```bash
cd ~/path/to/your/cuda/project
code gemm_naive.cu
```
This launches VS Code (remote mode) and opens add.cu.
<!-- screenshot: open_cuda_file -->

### 2. Edit your CUDA source file
Write or modify your CUDA code as usual.

For example:
```cuda
/* ============================================================================
   cuThermo pattern: MEMORY FALSE SHARING  -- the UNOPTIMIZED version.
   [paper Fig. 5(b); Sec. 6.1; Table 2 "GEMM / gemm_v00"; Table 4 "721.79%"]
   ----------------------------------------------------------------------------
   SOURCE
       Kernel `gemm_v00` is taken verbatim from
       https://github.com/leimao/CUDA-GEMM-Optimization
           src/00_non_coalesced_global_memory_access.cu
       which is reference [20] of the cuThermo paper and the origin of the
       paper's Listing 2.  Only the host harness below is new.

   THE INEFFICIENCY
       threadIdx.x feeds the C ROW index.  With blockDim 32x32 a warp is one
       row of threadIdx.x at fixed threadIdx.y, so the 32 lanes of a warp are
       32 DIFFERENT rows of C:

           A[C_row_idx*lda + k_idx]   lanes stride lda apart  -> 32 sectors
           C[C_row_idx*ldc + C_col_idx]  same                 -> 32 sectors
           B[k_idx*ldb + C_col_idx]   constant across a warp  -> broadcast

       B and C are where the FALSE SHARING shows up, and it is an ACROSS-WARP
       effect.  Warp w of the block has threadIdx.y = w, hence C_col_idx =
       blockIdx.y*32 + w.  So the 32 warps together touch 32 consecutive words
       = 4 sectors, and each 32-byte sector has its 8 words claimed by 8
       DIFFERENT warps:

           Sector | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |   sector total = 8
                    ^ each word one warp        ^ but the sector is 8x hotter

       Coalescing merges requests only WITHIN a warp, so those 8 words cost 8
       separate sector transactions instead of 1.  That is exactly the paper's
       Fig. 5(b), and Table 2 lists B and C of gemm_v00 as "False sharing".

   MEASURED (RTX A4500, N=1024, Nsight Compute)
       sectors per global LOAD request  : 16.53
       sectors per global STORE request : 32
       total global load sectors        : 17.4 M

   FIX: see false_sharing/gemm_opt.cu -- it is a two-line change.

   Build: nvcc -O3 -arch=sm_86 -lineinfo -o gemm_naive gemm_naive.cu
   ========================================================================= */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

/* ---- kernel, unchanged from leimao src/00_non_coalesced_global_memory_access.cu ---- */
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

/* leimao's launch config for v00: grid.x covers m, grid.y covers n. */
#define LAUNCH(A, B, Cm, N)                                                   \
    gemm_v00<float><<<dim3(((unsigned)(N)+31u)/32u, ((unsigned)(N)+31u)/32u),  \
                      dim3(32u, 32u)>>>((size_t)(N), (size_t)(N), (size_t)(N),\
                      1.0f, (A), (size_t)(N), (B), (size_t)(N), 0.0f,         \
                      (Cm), (size_t)(N))

#define KERNEL_TITLE "GEMM v00 -- non-coalesced / FALSE SHARING on B and C  [unoptimized]"
#define COMPARE_HINT "Compare with false_sharing/gemm_opt.cu (gemm_v01: two index lines swapped)."

/* ------------------------------------------------------------------ harness */
static void gemm_cpu(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int t = 0; t < N; t++) s += (double)A[(size_t)i*N+t] * B[(size_t)t*N+j];
            C[(size_t)i*N+j] = (float)s;
        }
}

static float time_it(const float *dA, const float *dB, float *dC, int N, int reps) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    LAUNCH(dA, dB, dC, N);                                  /* warm-up */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int r = 0; r < reps; r++) LAUNCH(dA, dB, dC, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / (float)reps;
}

int main(void) {
    printf("%s\n", KERNEL_TITLE);
    printf("blockDim 32x32; alpha=1, beta=0 so C = A*B\n");

    /* ---- (a) CORRECTNESS at N=512 against a CPU reference ---- */
    {
        int N = 512;
        size_t nb = (size_t)N*N*sizeof(float);
        float *hA = (float*)malloc(nb), *hB = (float*)malloc(nb);
        float *hC = (float*)malloc(nb), *hR = (float*)malloc(nb);
        srand(1);
        for (int i = 0; i < N*N; i++) { hA[i] = (float)((rand()%5)-2); hB[i] = (float)((rand()%5)-2); }
        gemm_cpu(hA, hB, hR, N);                     /* integer-valued => exact */

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, nb)); CUDA_CHECK(cudaMalloc(&dB, nb)); CUDA_CHECK(cudaMalloc(&dC, nb));
        CUDA_CHECK(cudaMemcpy(dA, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, nb));
        LAUNCH(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC, dC, nb, cudaMemcpyDeviceToHost));

        double max_err = 0.0; int bad = 0, first = -1;
        for (int i = 0; i < N*N; i++) {
            double d = fabs((double)hC[i] - (double)hR[i]);
            if (d > max_err) max_err = d;
            if (d > 1e-3) { if (first < 0) first = i; bad++; }
        }
        printf("\nCorrectness (N=%d, integer-valued so exact):\n", N);
        printf("  %s  (max error %.3e, %d/%d elements wrong)\n",
               bad == 0 ? "PASS" : "FAIL", max_err, bad, N*N);
        if (bad) printf("  first mismatch at %d: got %f, expected %f\n", first, hC[first], hR[first]);
        free(hA); free(hB); free(hC); free(hR);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    /* ---- (b) PARAMETER VARYING: N = 256 / 512 / 1024 ---- */
    printf("\nSize sweep:\n");
    for (int N : {256, 512, 1024}) {
        size_t nb = (size_t)N*N*sizeof(float);
        float *hA = (float*)malloc(nb), *hC = (float*)malloc(nb);
        for (int i = 0; i < N*N; i++) hA[i] = 1.0f;          /* all ones => C[i] == N */
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, nb)); CUDA_CHECK(cudaMalloc(&dB, nb)); CUDA_CHECK(cudaMalloc(&dC, nb));
        CUDA_CHECK(cudaMemcpy(dA, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hA, nb, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dC, 0, nb));
        float ms = time_it(dA, dB, dC, N, 10);
        CUDA_CHECK(cudaMemcpy(hC, dC, nb, cudaMemcpyDeviceToHost));
        bool ok = fabs((double)hC[0] - (double)N) < 1e-3;
        double gflop = 2.0*(double)N*N*N / 1e9;
        printf("  N=%4d : %9.3f ms  %7.1f GFLOP/s  (%s)\n",
               N, ms, gflop/(ms*1e-3), ok ? "ok" : "BAD");
        free(hA); free(hC);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    printf("\n%s\n", COMPARE_HINT);
    return 0;
}

/* Reference run — RTX A4500 (Ampere sm_86, CUDA 13.0), jli256-ub01:
GEMM v00 -- non-coalesced / FALSE SHARING on B and C  [unoptimized]
blockDim 32x32; alpha=1, beta=0 so C = A*B

Correctness (N=512, integer-valued so exact):
  PASS  (max error 0.000e+00, 0/262144 elements wrong)

Size sweep:
  N= 256 :     0.338 ms     99.4 GFLOP/s  (ok)
  N= 512 :     1.668 ms    160.9 GFLOP/s  (ok)
  N=1024 :    12.613 ms    170.3 GFLOP/s  (ok)

vs gemm_opt.cu at N=1024: 12.613 / 1.550 = 8.14x slower.
*/
```
You can get the code from the Github repository: [Parallel Demos](https://github.com/fruitfly1026/parallel-demos/blob/main/Demos/cuda/false_sharing/gemm_naive.cu)

### 3. Compile your CUDA program
In a terminal connected to the remote CUDA machine:
```bash
nvcc gemm_naive.cu -o gemm_naive 
```
This produces an executable named `gemm_naive`.

![nvcc gemm_naive.cu](images/nvcc_add.png)
<!-- screenshot: compile_cuda -->

### 4. Install the bundled `cuThermostat.so` (first-time setup)

In VS Code, press:

```css
Command + Shift + P   (macOS)
Ctrl + Shift + P      (Windows/Linux)
```

Then run:
```bash
cuThermo: Install Bundled cuThermostat.so to Workspace
```
This copies the profiler library into your project folder

![install cuthermo](images/install_cuthermo.png)


If you encountered the **warning** : "Open a folder/workspace first.", just run the command
```bash
code .
```
to open your current folder as the workspace (see screenshot below).
![open workspace](images/open_folder.png)


Now we can see the `Explorer` panel opened in the new VS Code window like follows, and we can try to install the cuThermo budle again:
![folder opened](images/folder_opened.png)


Once the `cuThermostat.so` is installed, we can see it appeared in the `Explorer` panel and the hint at the bottom right corner
![installed](images/installed.png)

### 5. Run cuThermo on the current target

Open the Command Palette again and run the command:

`cuThermo: Run in terminal`

![run in terminal](images/run_in_terminal.png)

and then select the executable:

![select executable](images/select_executable.png)

cuThermo will:

1. Verify that your executable (add) exists

2. Preload cuThermostat.so

3. Run the target under instrumentation

4. Generate an output text file (e.g., output_1234.txt)

5. Display the memory access heatmap

<!-- screenshot: run_on_current_target -->

![profile log](images/profile_log.png)

### 6. View the generated heatmap

Right click on the generated text file and choose "Open heatmap viewer"

![open heatmap](images/open_heatmap.png)

Finally, we can see the heatmap in the viewer:

![heatmap](images/heatmap.png)


## Miscellaneous

If you want to experience on my server, first email me your ssh **PUBLIC** (**NOT PRIVATE**) key

And then SSH to the server with username `jcui23` and hostname `eb2-3224-lin10.csc.ncsu.edu`, and your ssh **PRIVATE** key.


### 1. Config
You can add an item in your ssh config file like follows:

```YAML
Host 4090
  HostName eb2-3224-lin10.csc.ncsu.edu
  User jcui23
  IdentityFile /path/to/your/ssh-private-key
```
First, select the option `"Connect to Hosts"` after you click the connect icon on the left bottom corner

![connect to hosts](images/connect_to_host.png)


Next, choose the option `"Configure SSH Hosts..."`

![configure ssh hosts](images/configure_ssh_hosts.png)


After that, select the `config` file in the `.ssh` folder

![ssh config file](images/ssh_config.png)


In the config file, add the item like follows, but do remember to change the **IdentifyFile** path to yours, **NOT** mine.

**NO NEED TO MODIFY THE OTHER THREE LINES.**

![config item](images/config_item.png)

Now we can see the `4090` option when we click the connect button on the bottom left corner again

![4090 option](images/4090.png)

Hit it, when you are successfully connected, you can see the name appears on the bottom left corner.

![connected](images/connected.png)


### 2. Set CUDA
Once you get connected, you need to activate the CUDA environment to use the `nvcc` compiler to compile your CUDA program.

Open the `terminal` in VS Code and then run the following command in terminal to activate the CUDA environment:

```bash
source /scratch/setenvs/setcuda12.1.sh
```
![source cuda](images/source_cuda.png)

The following step is optional:
```bash
nvcc --version
```
The above command checks the compiler's version. Normal text will be printed out if you load the CUDA environment correctly.

![nvcc checking](images/nvcc.png)

Now you can use the nvcc to compile the CUDA source code (e.g. `main.cu`)
