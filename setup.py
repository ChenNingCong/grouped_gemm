import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. `torch` is required to install `grouped_gemm`.",) from e

# --- Compute Capability Determination ---
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")

if arch_list:
    # 1. If TORCH_CUDA_ARCH_LIST is set, use it to generate the flags.
    #    The arch_list is a string like "7.0 7.5 8.0+PTX".
    #    We need to parse it into 'compute_XX,code=sm_XX' pairs.
    device_capability_flags = []
    
    # Split the string by spaces and process each architecture string
    # E.g., '7.0', '7.5', '8.0+PTX'
    for arch_str in arch_list.split():
        # Clean up the string to get just the major.minor version (e.g., '70', '75', '80')
        # We replace the dot and remove the optional '+PTX' suffix.
        arch = arch_str.replace('.', '').split('+')[0]
        
        # The CUDA flag format is --generate-code=arch=compute_XX,code=sm_XX
        # where XX is the major/minor version (e.g., 70, 75, 80).
        flag = f"--generate-code=arch=compute_{arch},code=sm_{arch}"
        device_capability_flags.append(flag)

else:
    # 2. If TORCH_CUDA_ARCH_LIST is NOT set, query the current device capability.
    if torch.cuda.is_available():
        # Get the capability (e.g., (8, 0)) and format it as '80'
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        
        # Create a list with the single flag for the current device.
        flag = f"--generate-code=arch=compute_{arch},code=sm_{arch}"
        device_capability_flags = [flag]
    else:
        # Handle case where CUDA is not available or device capability can't be found
        print("Warning: CUDA not available. Compiling without device-specific architecture flags.")
        device_capability_flags = []
# --- NVCC Flags Setup ---
cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17,
    "-O3",
]
nvcc_flags.extend(device_capability_flags)
print(nvcc_flags)
if os.environ.get("GROUPED_GEMM_CUTLASS", "0") == "1":
    nvcc_flags.extend(["-DGROUPED_GEMM_CUTLASS"])

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        ["csrc/ops.cu", "csrc/grouped_gemm.cu", "csrc/grouped_gemm_sm89.cu"],
        include_dirs = [
            f"{cwd}/third_party/cutlass/include/",
            f"{cwd}/csrc"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": nvcc_flags,
        }
    )
]

extra_deps = {}

extra_deps['dev'] = [
    'absl-py',
]

extra_deps['all'] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="grouped_gemm",
    version="0.3.0",
    author="Trevor Gale",
    author_email="tgale@stanford.edu",
    description="Grouped GEMM",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/tgale06/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    extras_require=extra_deps,
)
