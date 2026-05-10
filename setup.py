from __future__ import annotations

import sys

from setuptools import Extension, setup


is_windows = sys.platform == "win32"

setup(
    name="rinha-native",
    version="0.1.0",
    ext_modules=[
        Extension(
            "rinha_native",
            ["rinha_native.c"],
            extra_compile_args=["/O2"] if is_windows else ["-O3"],
            libraries=[] if is_windows else ["m"],
        )
    ],
)
