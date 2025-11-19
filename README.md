# cuThermo

**cuThermo** is a Visual Studio Code extension that profiles CUDA applications and visualizes GPU memory-access heatmaps. It integrates with your CUDA toolchain and the bundled `cuThermostat.so` profiler to make GPU performance analysis easier, more intuitive, and fully interactive inside VS Code.

---

## 🚀 Features

- **Run GPU profiling directly in VS Code**  
  Invoke *cuThermo: Run on Current Target* to execute the selected CUDA binary and collect memory-access traces.

- **Automatic heatmap visualization**  
  Profiling results are parsed and displayed in a custom WebView panel, showing a live heatmap of memory sector activity.

- **Duplicate-row collapsing**  
  The heatmap applies intelligent preprocessing (same logic as your web-based heatmap tool) to compress repetitive frequency rows for a more readable visualization.

- **Environment validation**  
  Includes a command to check CUDA toolkit, `nvcc`, GPU availability, and whether `cuThermostat.so` is correctly placed.

- **Configurable executable selection**  
  Users can specify any CUDA executable, not just `a.out`.

---

## 📦 Requirements

To use cuThermo effectively:

- **CUDA Toolkit** (nvcc available in PATH)  
- **NVIDIA GPU + drivers**  
- **cuThermostat.so**  
  - Either bundled or installed into the workspace  
  - The extension includes a command to copy/verify the shared library

Optional but recommended:

- VS Code C++ extension  
- NVIDIA Nsight Systems (for advanced users)

---

## Install cuThermo for the course

1. Download the latest `.vsix` from the Releases page:
   - https://github.com/CuiJinku/cuthermo-vscode/releases
2. In VS Code: Extensions → `…` menu → *Install from VSIX…*
3. Select the downloaded `JinkuCui.cuthermo-0.0.3.vsix`.

---

## ⚙️ Extension Settings

This extension contributes the following settings:

### `cuthermo.execPath`
Path to the target executable (e.g. `./mykernel`, `build/add`, etc.).

### `cuthermo.thermostatPath`
Path to `cuThermostat.so` if you want to override the default bundled version.

### `cuthermo.autoOpenHeatmap`
Whether to automatically open the heatmap window after each profiling run.

---

## 🧩 Commands

| Command | Description |
|--------|-------------|
| **cuThermo: Run on Current Target** | Profiles the selected CUDA executable and generates a heatmap. |
| **cuThermo: Install cuThermostat.so to Workspace** | Copies the bundled shared library into your workspace. |
| **cuThermo: Check Environment** | Verifies CUDA, GPU, and toolchain setup. |

---

## 🧪 Known Issues

- Heatmap panel may not resize correctly on very small displays.  
- Some unusual workspace structures may cause path resolution inconsistencies.  
- Requires CUDA-compatible NVIDIA hardware — no support for AMD/ROCm yet.

Please report issues on the GitHub repo.

---

## 📝 Release Notes

### 0.1.0
- Initial public release
- Heatmap visualization
- Profile-run integration
- Environment checking
- Executable path configuration

---

## 🔗 More Information

- Documentation for developing VS Code extensions:  
  https://code.visualstudio.com/api

- cuThermo GitHub repository (if public):  
  *(add link here)*

---

**Enjoy profiling with cuThermo!**
