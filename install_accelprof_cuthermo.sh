#!/usr/bin/env bash

# VERSION: 2026-09-04-v7-clean-examples
set -euo pipefail

# ============================================================
# AccelProf cuThermo Branch Installation Script
# ============================================================
#
# This script:
#   1. Clones the AccelProf cuThermo branch WITHOUT --recursive.
#   2. Initializes the normal submodules.
#   3. Replaces sanalyzer and nv-compute with the modified
#      FlagZhao/cuThermo versions.
#   4. Initializes sanalyzer's internal submodules recursively.
#   5. Verifies critical header files.
#   6. Creates a Conda environment and installs PyTorch.
#   7. Builds AccelProf components in dependency order.
#
# IMPORTANT:
# Do NOT replace the clone command below with:
#
#   git clone --recursive ...
#
# The upstream nv-compute submodule references a commit that is
# not available:
#
#   3b6a80ffdab182ea3fae959c4023dc7a5aa61b1b
#
# ============================================================

REPO_URL="https://github.com/FlagZhao/AccelProf.git"
REPO_BRANCH="cuThermo"

SANALYZER_URL="https://github.com/FlagZhao/sanalyzer.git"
NV_COMPUTE_URL="https://github.com/FlagZhao/nv-compute.git"

WORKSPACE="${WORKSPACE:-$HOME/workspace}"
REPO_DIR="$WORKSPACE/AccelProf"

log() {
    printf '\n\033[1;34m[AccelProf Setup]\033[0m %s\n' "$1"
}

error() {
    printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$1" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || error "Required command '$1' was not found in PATH."
}

verify_file() {
    local description="$1"
    local path="$2"

    if [[ -f "$path" ]]; then
        printf '  [OK] %s: %s\n' "$description" "$path"
    else
        printf '  [MISSING] %s: %s\n' "$description" "$path" >&2
        return 1
    fi
}

# ------------------------------------------------------------
# 0. Basic checks
# ------------------------------------------------------------

require_command git
require_command find
require_command conda
require_command nvcc
require_command gcc
require_command g++
require_command curl

GCC_MAJOR="$(gcc -dumpfullversion -dumpversion | cut -d. -f1)"
GXX_MAJOR="$(g++ -dumpfullversion -dumpversion | cut -d. -f1)"

if (( GCC_MAJOR < 11 )); then
    error "GCC 11 or newer is required for the C++20 build. Found: $(gcc --version | head -n1)"
fi

if (( GXX_MAJOR < 11 )); then
    error "G++ 11 or newer is required for the C++20 build. Found: $(g++ --version | head -n1)"
fi

log "Compiler check passed."
printf '  %s\n' "$(gcc --version | head -n1)"
printf '  %s\n' "$(g++ --version | head -n1)"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [[ -d "$REPO_DIR/.git" ]]; then
    log "Existing AccelProf checkout found at $REPO_DIR. Reusing it."
    cd "$REPO_DIR"
elif [[ -e "$REPO_DIR" ]]; then
    error "$REPO_DIR exists but is not a Git repository. Rename or remove it before rerunning this script."
else
    # ------------------------------------------------------------
    # 1. Clone main repository WITHOUT --recursive
    # ------------------------------------------------------------

    log "Cloning AccelProf branch '$REPO_BRANCH' into $REPO_DIR ..."
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ------------------------------------------------------------
# 2. Initialize normal submodules
# ------------------------------------------------------------

log "Initializing standard AccelProf submodules ..."

git submodule update --init \
    amd-rocm \
    docs \
    nv-nvbit \
    tensor_scope \
    third_party/libbacktrace \
    third_party/pybind11

# sanalyzer and nv-compute are intentionally NOT initialized from
# the parent repository. They are replaced below with modified
# cuThermo versions.

# ------------------------------------------------------------
# 3. Manually clone modified submodules
# ------------------------------------------------------------

log "Preparing modified sanalyzer and nv-compute repositories ..."

if [[ -d sanalyzer/.git ]]; then
    log "Existing sanalyzer checkout found. Reusing it."
else
    rm -rf sanalyzer
    git clone -b "$REPO_BRANCH" "$SANALYZER_URL" sanalyzer
fi

if [[ -d nv-compute/.git ]]; then
    log "Existing nv-compute checkout found. Reusing it."
else
    rm -rf nv-compute
    git clone -b "$REPO_BRANCH" "$NV_COMPUTE_URL" nv-compute
fi

# ------------------------------------------------------------
# 4. Initialize sanalyzer internal submodules
# ------------------------------------------------------------

log "Initializing sanalyzer internal submodules ..."

(
    cd sanalyzer
    git submodule update --init --recursive
)

# ------------------------------------------------------------
# 5. Verify critical header files before building
# ------------------------------------------------------------

log "Verifying critical source/header files ..."

prebuild_ok=1

verify_file "gpu_patch.h" \
    "$(find nv-compute -name gpu_patch.h -print -quit)" || prebuild_ok=0

verify_file "nvbit_common.h" \
    "$(find nv-nvbit -name nvbit_common.h -print -quit)" || prebuild_ok=0

verify_file "cpp_trace.h" \
    "$(find sanalyzer/cpp_trace -name cpp_trace.h -print -quit)" || prebuild_ok=0

verify_file "py_frame.h" \
    "$(find sanalyzer/py_frame -name py_frame.h -print -quit)" || prebuild_ok=0

if [[ "$prebuild_ok" -ne 1 ]]; then
    error "One or more required files are missing. Check the submodule checkout before building."
fi

# ------------------------------------------------------------
# 6. Set up Python environment
# ------------------------------------------------------------

log "Setting up Conda environment ..."

CONDA_BASE="$(conda info --base)"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
    error "Could not find Conda shell initialization script at: $CONDA_SH"
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

ENV_NAME="${ENV_NAME:-accelprof}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' already exists. Reusing it."
else
    log "Creating Conda environment '$ENV_NAME' with Python 3.11 ..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

log "Active Python environment:"
printf '  python: %s\n' "$(command -v python)"
printf '  version: %s\n' "$(python --version 2>&1)"

log "Checking PyTorch installation ..."

if python -c "import torch" >/dev/null 2>&1; then
    log "PyTorch is already installed in Conda environment '$ENV_NAME'. Reusing it."
else
    log "Installing PyTorch and torchvision ..."
    python -m pip install --upgrade pip
    python -m pip install torch torchvision
fi

log "Verifying PyTorch installation and C++ headers ..."

TORCH_INCLUDE_DIR="$(python - <<'PY'
import torch
from pathlib import Path
print(Path(torch.__file__).resolve().parent / "include")
PY
)"

TORCH_ATEN_HEADER="$TORCH_INCLUDE_DIR/ATen/record_function.h"

if [[ ! -f "$TORCH_ATEN_HEADER" ]]; then
    error "PyTorch is installed, but required C++ header is missing: $TORCH_ATEN_HEADER"
fi

python - <<'PY'
import torch
from torch.utils.cpp_extension import include_paths

print("PyTorch version:", torch.__version__)
print("PyTorch location:", torch.__file__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch C++ include paths:")
for path in include_paths():
    print("  ", path)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

printf '  [OK] Required PyTorch header: %s\n' "$TORCH_ATEN_HEADER"

# ------------------------------------------------------------
# 7. Build AccelProf components explicitly
# ------------------------------------------------------------

export ACCEL_PROF_DIR="$REPO_DIR"
export DEBUG="${DEBUG:-0}"

log "Building AccelProf components explicitly ..."
printf '  ACCEL_PROF_DIR=%s\n' "$ACCEL_PROF_DIR"
printf '  DEBUG=%s\n' "$DEBUG"

mkdir -p "$ACCEL_PROF_DIR/lib"

# 7.1 Build libbacktrace if needed.
log "Building libbacktrace (if needed) ..."
cd "$ACCEL_PROF_DIR/sanalyzer/cpp_trace"

if [[ ! -d "$ACCEL_PROF_DIR/build/backtrace" ]]; then
    # shellcheck disable=SC1091
    source "$ACCEL_PROF_DIR/bin/utils/build_libbacktrace"
else
    log "libbacktrace build directory already exists; skipping rebuild."
fi

# 7.2 Build cpp_trace.
log "Building cpp_trace ..."
cd "$ACCEL_PROF_DIR/sanalyzer/cpp_trace"

make -j install \
    DEBUG="$DEBUG" \
    BACKTRACE_DIR="$ACCEL_PROF_DIR/build/backtrace" \
    INSTALL_DIR="$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace"

# 7.3 Build py_frame.
log "Building py_frame ..."
cd "$ACCEL_PROF_DIR/sanalyzer/py_frame"

make -j install \
    DEBUG="$DEBUG" \
    PYBIND11_DIR="$ACCEL_PROF_DIR/third_party/pybind11" \
    INSTALL_DIR="$ACCEL_PROF_DIR/build/sanalyzer/py_frame"

# 7.4 Build sanalyzer.
log "Building sanalyzer ..."
cd "$ACCEL_PROF_DIR/sanalyzer"

make -j install \
    DEBUG="$DEBUG" \
    SANITIZER_TOOL_DIR="$ACCEL_PROF_DIR/nv-compute" \
    NV_NVBIT_DIR="$ACCEL_PROF_DIR/nv-nvbit" \
    CPP_TRACE_DIR="$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace" \
    PY_FRAME_DIR="$ACCEL_PROF_DIR/build/sanalyzer/py_frame" \
    INSTALL_DIR="$ACCEL_PROF_DIR/build/sanalyzer"

# 7.5 Build tensor_scope.
log "Building tensor_scope ..."

# tensor_scope's Makefile queries the active Python environment for
# PyTorch include paths, so the Conda environment must still be active here.
python -c "import torch" || error "PyTorch is not importable from the active Python environment."

TENSOR_SCOPE_MAKEFILE="$ACCEL_PROF_DIR/tensor_scope/Makefile"

if [[ ! -f "$TENSOR_SCOPE_MAKEFILE" ]]; then
    error "tensor_scope Makefile not found: $TENSOR_SCOPE_MAKEFILE"
fi

# Current PyTorch C++ headers require C++20. The cuThermo branch may still
# specify -std=c++17, so patch it automatically before compilation.
if grep -q -- "-std=c++17" "$TENSOR_SCOPE_MAKEFILE"; then
    log "Updating tensor_scope compiler standard from C++17 to C++20 ..."
    sed -i 's/-std=c++17/-std=c++20/g' "$TENSOR_SCOPE_MAKEFILE"
fi

if ! grep -q -- "-std=c++20" "$TENSOR_SCOPE_MAKEFILE"; then
    error "tensor_scope Makefile does not contain -std=c++20 after patching."
fi

printf '  [OK] tensor_scope uses C++20\n'
grep -n -- "-std=c++20" "$TENSOR_SCOPE_MAKEFILE" | head -n 3 || true

cd "$ACCEL_PROF_DIR/tensor_scope"

make -j install \
    DEBUG="$DEBUG" \
    INSTALL_DIR="$ACCEL_PROF_DIR/build/tensor_scope"

if [[ ! -f "$ACCEL_PROF_DIR/build/tensor_scope/include/torch_scope.h" ]]; then
    error "tensor_scope build did not produce build/tensor_scope/include/torch_scope.h"
fi

printf '  [OK] %s\n' "$ACCEL_PROF_DIR/build/tensor_scope/include/torch_scope.h"

# 7.6 Build nv-compute and copy its shared libraries.
log "Building nv-compute ..."
cd "$ACCEL_PROF_DIR/nv-compute"

make -j \
    DEBUG="$DEBUG" \
    SANALYZER_DIR="$ACCEL_PROF_DIR/build/sanalyzer" \
    TORCH_SCOPE_DIR="$ACCEL_PROF_DIR/build/tensor_scope" \
    PATCH_SRC_DIR="$ACCEL_PROF_DIR/nv-compute/gpu_src"

if compgen -G "$ACCEL_PROF_DIR/nv-compute/lib/*.so" > /dev/null; then
    cp "$ACCEL_PROF_DIR"/nv-compute/lib/*.so "$ACCEL_PROF_DIR/lib/"
else
    error "nv-compute build completed but no .so files were found in nv-compute/lib/."
fi

# 7.7 Build nv-nvbit and copy its shared libraries.
log "Building nv-nvbit ..."
cd "$ACCEL_PROF_DIR/nv-nvbit"

make -j \
    DEBUG="$DEBUG" \
    SANALYZER_DIR="$ACCEL_PROF_DIR/build/sanalyzer" \
    TORCH_SCOPE_DIR="$ACCEL_PROF_DIR/build/tensor_scope"

if compgen -G "$ACCEL_PROF_DIR/nv-nvbit/lib/*.so" > /dev/null; then
    cp "$ACCEL_PROF_DIR"/nv-nvbit/lib/*.so "$ACCEL_PROF_DIR/lib/"
else
    error "nv-nvbit build completed but no .so files were found in nv-nvbit/lib/."
fi

cd "$ACCEL_PROF_DIR"

# ------------------------------------------------------------
# 8. Verify important build outputs
# ------------------------------------------------------------

log "Verifying build outputs ..."

build_ok=1

if [[ -d build/backtrace ]]; then
    printf '  [OK] build/backtrace/\n'
else
    printf '  [MISSING] build/backtrace/\n' >&2
    build_ok=0
fi

for path in \
    build/sanalyzer/lib/libsanalyzer.so \
    build/sanalyzer/include/sanalyzer.h \
    build/sanalyzer/cpp_trace/include/cpp_trace.h \
    build/sanalyzer/py_frame/include/py_frame.h \
    lib/libcompute_sanitizer.so \
    lib/libnv-nvbit.so
do
    if [[ -f "$path" ]]; then
        printf '  [OK] %s\n' "$path"
    else
        printf '  [MISSING] %s\n' "$path" >&2
        build_ok=0
    fi
done

if [[ "$build_ok" -ne 1 ]]; then
    error "The build finished, but one or more required outputs are missing."
fi

log "AccelProf build completed successfully."

log "Runtime libraries available in $ACCEL_PROF_DIR/lib:"
find "$ACCEL_PROF_DIR/lib" -maxdepth 1 -type f -name "*.so" -printf '  %f\n' | sort

# ------------------------------------------------------------
# 9. Install the AccelProf VS Code extension
# ------------------------------------------------------------

VSIX_PATH="$ACCEL_PROF_DIR/third_party/accelprof-vscode/accelprof-vscode-0.1.0.vsix"

log "Checking AccelProf VS Code extension ..."

if [[ ! -f "$VSIX_PATH" ]]; then
    error "VS Code extension package was not found: $VSIX_PATH"
fi

if command -v code >/dev/null 2>&1; then
    log "Installing AccelProf VS Code extension ..."
    code --install-extension "$VSIX_PATH" --force
    log "VS Code extension installation completed."
else
    printf '\n\033[1;33m[WARNING]\033[0m VS Code CLI command "code" was not found.\n'
    printf 'The AccelProf build is complete, but the VS Code extension was not installed automatically.\n'
    printf 'Install this VSIX manually from VS Code using "Extensions: Install from VSIX...":\n'
    printf '  %s\n\n' "$VSIX_PATH"
fi

# ------------------------------------------------------------
# 10. Download, compile, and profile CUDA heatmap examples
# ------------------------------------------------------------

EXAMPLES_DIR="$ACCEL_PROF_DIR/examples/false_sharing"
NAIVE_SRC="$EXAMPLES_DIR/gemm_naive.cu"
OPT_SRC="$EXAMPLES_DIR/gemm_opt.cu"
NAIVE_EXE="$EXAMPLES_DIR/gemm_naive"
OPT_EXE="$EXAMPLES_DIR/gemm_opt"

NAIVE_URL="https://raw.githubusercontent.com/fruitfly1026/parallel-demos/main/Demos/cuda/false_sharing/gemm_naive.cu"
OPT_URL="https://raw.githubusercontent.com/fruitfly1026/parallel-demos/main/Demos/cuda/false_sharing/gemm_opt.cu"

log "Preparing CUDA heatmap examples ..."
mkdir -p "$EXAMPLES_DIR"

# Download sources only if they are missing.
if [[ -f "$NAIVE_SRC" ]]; then
    log "Source already exists. Reusing: $NAIVE_SRC"
else
    log "Downloading gemm_naive.cu ..."
    curl -fL "$NAIVE_URL" -o "$NAIVE_SRC"
fi

if [[ -f "$OPT_SRC" ]]; then
    log "Source already exists. Reusing: $OPT_SRC"
else
    log "Downloading gemm_opt.cu ..."
    curl -fL "$OPT_URL" -o "$OPT_SRC"
fi

# Compile only if the executable is missing or older than its source file.
if [[ -x "$NAIVE_EXE" && "$NAIVE_EXE" -nt "$NAIVE_SRC" ]]; then
    log "Executable is up to date. Reusing: $NAIVE_EXE"
else
    log "Compiling gemm_naive.cu with CUDA line information ..."
    nvcc -lineinfo \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_89,code=compute_89 \
        -o "$NAIVE_EXE" "$NAIVE_SRC"
fi

if [[ -x "$OPT_EXE" && "$OPT_EXE" -nt "$OPT_SRC" ]]; then
    log "Executable is up to date. Reusing: $OPT_EXE"
else
    log "Compiling gemm_opt.cu with CUDA line information ..."
    nvcc -lineinfo \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_89,code=compute_89 \
        -o "$OPT_EXE" "$OPT_SRC"
fi

if [[ ! -x "$ACCEL_PROF_DIR/bin/accelprof" ]]; then
    error "AccelProf executable was not found or is not executable: $ACCEL_PROF_DIR/bin/accelprof"
fi

# Keep the teaching directory simple: remove previous heatmap results/logs
# before generating a fresh pair.
log "Removing previous heatmap example outputs ..."

find "$EXAMPLES_DIR" -maxdepth 1 -type d \
    \( -name 'heatmap_gemm_naive_*' -o -name 'heatmap_gemm_opt_*' \) \
    -print -exec rm -rf {} +

rm -f \
    "$EXAMPLES_DIR/gemm_naive.accelprof.log" \
    "$EXAMPLES_DIR/gemm_opt.accelprof.log"

# Run each profile from the example directory so generated profiler output
# stays next to the corresponding source and executable.
log "Generating heatmap for the unoptimized GEMM example ..."
(
    cd "$EXAMPLES_DIR"
    "$ACCEL_PROF_DIR/bin/accelprof" -v -t heatmap_analysis ./gemm_naive
)

log "Generating heatmap for the optimized GEMM example ..."
(
    cd "$EXAMPLES_DIR"
    "$ACCEL_PROF_DIR/bin/accelprof" -v -t heatmap_analysis ./gemm_opt
)

log "CUDA heatmap examples completed."
printf '  Source directory: %s\n' "$EXAMPLES_DIR"
printf '  Unoptimized executable: %s\n' "$NAIVE_EXE"
printf '  Optimized executable:   %s\n' "$OPT_EXE"

printf '\033[1;36m'

cat <<'EOF'

Heatmap usage
-------------
Two CUDA examples have been downloaded to:

    $ACCEL_PROF_DIR/examples/false_sharing/

They are:

    gemm_naive.cu   - unoptimized GEMM
    gemm_opt.cu     - optimized GEMM

The installation script compiles both programs and runs:

    accelprof -v -t heatmap_analysis <executable>

Students can use the AccelProf VS Code extension to open the generated
heatmap data and compare the unoptimized and optimized implementations.

When compiling additional CUDA programs for heatmap/source-line analysis,
remember to include CUDA line information. For an NVIDIA RTX 4090
(Ada, compute capability 8.9), use:

    nvcc -lineinfo \
      -gencode arch=compute_89,code=sm_89 \
      -gencode arch=compute_89,code=compute_89 \
      -o <executable> <source.cu>

Then run the heatmap tool with:

    accelprof -v -t heatmap_analysis <executable>

If AccelProf is not on PATH, use:

    $ACCEL_PROF_DIR/bin/accelprof -v -t heatmap_analysis <executable>

NOTE:
The sm_89 / compute_89 flags above target NVIDIA Ada GPUs such as the RTX 4090.
Use the appropriate compute capability if your assigned cluster GPU uses a
different architecture.

EOF

printf '\033[0m'
