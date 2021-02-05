from pathlib import Path
from distutils.core import setup

import numpy
from Cython.Build import cythonize

_DIR = Path(__file__).parent

setup(
    name="monotonic_align",
    ext_modules=cythonize(str(_DIR / "core.pyx")),
    include_dirs=[numpy.get_include()],
)
