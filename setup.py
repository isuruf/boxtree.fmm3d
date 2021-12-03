from setuptools import find_namespace_packages

from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from glob import glob
import os

# generate fortran.pyf using
# f2py -m fortran -h boxtree/fmm3d/fortran.pyf FMM3D/src/Laplace/lfmm3d.f FMM3D/src/Helmholtz/hfmm3d.f

USE_EXTERNAL_FMM3D = os.getenv("BOXTREE_FMM3D_USE_EXTERNAL", "0") == "1"

sources = ["boxtree/fmm3d/fortran.pyf"]
libraries = []
library_dirs = []

if not USE_EXTERNAL_FMM3D:
    exclude_sources = [
        'FMM3D/src/Common/tree_lr_3d.f',
        'FMM3D/src/Helmholtz/helmkernels.f',
        'FMM3D/src/Helmholtz/helmkernels_dr.f',
        'FMM3D/src/Laplace/lapkernels_dr.f',
        'FMM3D/src/Helmholtz/hwts3.f',
        ] + glob("FMM3D/src/**/*_fast.f")
    sources += glob("FMM3D/src/Laplace/*.f") + glob("FMM3D/src/Helmholtz/*.f")
    sources += glob("FMM3D/src/Common/*.f")
    sources = list(set(sources) - set(exclude_sources))
else:
    libraries += ["fmm3d"]
    library_dirs = ["FMM3D/lib-static"]

print(sources)
#1/0

exts = [Extension(
    name="boxtree.fmm3d.fortran",
    sources=sources,
    f2py_options=["only:", "lfmm3dmain", "hfmm3dmain", ":"],
    libraries=libraries,
    library_dirs=library_dirs,
    extra_link_args=["-fopenmp"],
    extra_f77_compile_args=["-std=legacy", "-fopenmp", "-fPIC",
        "-O3", "-funroll-loops", "-Wno-unused-variable", "-Wno-tabs",
        "-Wno-conversion", "-Wno-maybe-uninitialized", "-Wno-unused-dummy-argument",
        "-Wno-unused-label"],
    )]

setup(
    name="boxtree.fmm3d",
    ext_modules=exts,
    packages=find_namespace_packages(include=["boxtree.*"]),
)
