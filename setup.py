import glob
import platform
import subprocess
from pathlib import Path

import setuptools
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, naive_recompile
from setuptools.command.build_ext import build_ext

extra_compile_args = []
extra_link_args = []

# OpenMP support + optimization flags
if platform.system() == "Linux":
    extra_compile_args += ["-fopenmp", "-O2", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    extra_link_args += ["-fopenmp"]
elif platform.system() == "Darwin":
    # Homebrew installs libomp headers/libs outside the default search paths
    try:
        libomp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        extra_compile_args += [f"-I{libomp_prefix}/include"]
        extra_link_args += [f"-L{libomp_prefix}/lib"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    extra_compile_args += ["-Xpreprocessor", "-fopenmp", "-O2", "-DNDEBUG", "-ffast-math", "-funroll-loops"]
    extra_link_args += ["-lomp"]
elif platform.system() == "Windows":
    extra_compile_args += ["/openmp", "/O2", "/fp:fast", "/DNDEBUG"]

# Sources that must keep strict IEEE floating-point semantics.
# src/forecast_proportions.cpp ports NumPy's pairwise summation and must stay
# bit-identical to the pure-Python reference traversal, including NaN/inf
# propagation: under -ffast-math (finite-math-only) the kernel's
# `std::abs(child_sum) < 1e-8` check is allowed to assume no NaNs, so a NaN
# child sum silently takes the equal-split branch and produces finite output
# where the reference propagates NaN. The in-file FP pragmas cannot fully undo
# a driver-level -ffast-math on every toolchain, so these files are excluded
# from fast-math here; the pragmas remain as defense-in-depth only.
STRICT_FP_SOURCES = {"forecast_proportions.cpp"}


class strict_fp_build_ext(build_ext):
    """build_ext that compiles ``STRICT_FP_SOURCES`` without fast-math.

    setuptools has no per-source flag support, so this hooks the per-file
    compile step instead: on unix-like compilers it wraps ``_compile`` and
    appends ``-fno-fast-math`` (the later flag overrides the earlier
    ``-ffast-math`` on both gcc and clang); on MSVC, whose ``compile`` issues
    one ``spawn`` per source, it wraps ``spawn`` and appends ``/fp:precise``
    (the later flag overrides the earlier ``/fp:fast``).
    """

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            original_spawn = self.compiler.spawn

            def spawn(cmd, *args, **kwargs):
                if any(Path(str(part)).name in STRICT_FP_SOURCES for part in cmd):
                    cmd = [*cmd, "/fp:precise"]
                return original_spawn(cmd, *args, **kwargs)

            self.compiler.spawn = spawn
        else:
            original_compile = self.compiler._compile

            def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if Path(src).name in STRICT_FP_SOURCES:
                    extra_postargs = [*extra_postargs, "-fno-fast-math"]
                return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

            self.compiler._compile = _compile
        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        name="hierarchicalforecast._lib",
        sources=glob.glob("src/*.cpp"),
        include_dirs=["external_libs/eigen"],
        cxx_std=20,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

ParallelCompile(
    "CMAKE_BUILD_PARALLEL_LEVEL", needs_recompile=naive_recompile
).install()

setuptools.setup(ext_modules=ext_modules, cmdclass={"build_ext": strict_fp_build_ext})
