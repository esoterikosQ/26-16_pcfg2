"""C++ CKY 엔진(pcfg._ckycpp) 빌드.

    python setup.py build_ext --inplace

- C++17, -O3. 비트 단위 동치(Python 엔진과 같은 로그확률)를 위해 -ffast-math 등
  부동소수점 재결합·축약을 허용하는 옵션은 쓰지 않고 -ffp-contract=off 를 명시한다.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "pcfg._ckycpp",
        ["pcfg/_ckycpp.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-ffp-contract=off"],
    ),
]

setup(
    name="pcfg",
    version="0.1.0",
    packages=["pcfg"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
