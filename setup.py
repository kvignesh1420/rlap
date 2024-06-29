"""Setup for pip package.
Adapted from : https://github.com/tensorflow/io/blob/master/setup.py
"""

import os
import glob

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension,
)

library_name = "rlap"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    extension = CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-fvisibility=hidden",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-DEIGEN_MAX_ALIGN_BYTES=64",
            # "-std=c++14",
            "-DEIGEN_MPL2_ONLY",
            "-msse4.2",
            "-mavx",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cc")))
    # eigen_include_dir = "/usr/local/include/eigen3"
    eigen_include_dir = "third_party/eigen-3.4.0"
    rlap_include_dir = "rlap/csrc"

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=[eigen_include_dir, rlap_include_dir],
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    description="rlap",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kvignesh1420/rlap",
    author="Vignesh Kothapalli",
    author_email="k.vignesh1420@gmail.com",
    packages=find_packages(),
    ext_modules=get_extensions(),
    python_requires=">=3.8, <3.12",
    install_requires=["torch"],
    project_urls={
        "Source": "https://github.com/kvignesh1420/rlap",
        "Bug Reports": "https://github.com/kvignesh1420/rlap/issues",
        "Documentation": "https://github.com/kvignesh1420/rlap",
    },
    zip_safe=False,
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
