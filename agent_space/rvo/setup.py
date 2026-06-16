"""Minimum-effort build: one C++ file via torch.utils.cpp_extension (no CMake).

pip install -e . --no-build-isolation
"""

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="rvo",
    version="0.0.1",
    packages=["rvo"],
    ext_modules=[CppExtension(name="rvo._C", sources=["csrc/rvo.cpp"])],
    cmdclass={"build_ext": BuildExtension},
    # autoload on `import torch` so users never type `import rvo`:
    entry_points={"torch.backends": ["rvo = rvo:_autoload"]},
)
