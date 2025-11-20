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

For demonstration, we use `add.cu` for the tutorial.
In a terminal on the remote machine:

```bash
cd ~/path/to/your/cuda/project
code add.cu
```
This launches VS Code (remote mode) and opens add.cu.
<!-- screenshot: open_cuda_file -->

### 2. Edit your CUDA source file
Write or modify your CUDA code as usual.

For example:
```cuda
#include <iostream>
#include <math.h>
 
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
   y[i] = x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<20;
 float *x, *y;
 
 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Run kernel on 1M elements on the GPU
 add<<<1, 1>>>(N, x, y);
 
 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();
 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFree(x);
 cudaFree(y);
  return 0;
}
```
You can get the code from the tutorial: [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### 3. Compile your CUDA program
In a terminal connected to the remote CUDA machine:
```bash
nvcc add.cu -o add
```
This produces an executable named `add`.

![nvcc add.cu](images/nvcc_add.png)
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
