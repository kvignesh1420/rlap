
"""Setup for pip package.
Adapted from : https://github.com/tensorflow/io/blob/master/setup.py
"""

import os
import sys
import shutil
import tempfile
import fnmatch
import setuptools

here = os.path.abspath(os.path.dirname(__file__))


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

project = "rlap"
project_rootpath = project
datapath = None

if "--data" in sys.argv:
    data_idx = sys.argv.index("--data")
    datapath = sys.argv[data_idx + 1]
    sys.argv.remove("--data")
    sys.argv.pop(data_idx)
else:
    datapath = os.environ.get("RLAP_DATAPATH")

if (datapath is not None) and ("bdist_wheel" in sys.argv):
    rootpath = tempfile.mkdtemp()
    print(f"setup.py - create {rootpath} and copy {project_rootpath} data files")
    for rootname, _, filenames in os.walk(os.path.join(datapath, project_rootpath)):
        if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(
            rootname, "*runfiles*"
        ):
            for filename in [
                f
                for f in filenames
                if fnmatch.fnmatch(f, "*.so") or fnmatch.fnmatch(f, "*.py")
            ]:
                src = os.path.join(rootname, filename)
                dst = os.path.join(
                    rootpath,
                    os.path.relpath(os.path.join(rootname, filename), datapath),
                )
                print(f"setup.py - copy {src} to {dst}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
    sys.argv.append("--bdist-dir")
    sys.argv.append(rootpath)

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

class BinaryDistribution(setuptools.dist.Distribution):
    def has_ext_modules(self):
        return True

setuptools.setup(
    name=project,
    version="0.1.0",
    description="rlap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kvignesh1420/rlap",
    author="Vignesh Kothapalli",
    author_email="k.vignesh1420@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=setuptools.find_packages(where=".", exclude=[]),
    python_requires=">=3.8, <3.12",
    install_requires=install_requires,
    package_data={
        ".": ["*.so"],
    },
    project_urls={
        "Source": "https://github.com/kvignesh1420/rlap",
        "Bug Reports": "https://github.com/kvignesh1420/rlap/issues",
        "Documentation": "https://github.com/kvignesh1420/rlap",
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)