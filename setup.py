#!/usr/bin/env python
from setuptools import setup

setup(
    name="rlrl",
    version="1.0.0",
    install_requires=[
        "torch",
        "gym",
        "numpy",
    ],
    extras_require={
        "dev": [
            "wandb",
            "mujoco-py",
            "black",
            "flake8",
            "isort",
            "moviepy",
            "imageio",
        ]
    },
    description="Library for RL research",
    author="hisaki",
    packages=["rlrl"],
)
