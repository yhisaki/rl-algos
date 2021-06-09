from setuptools import setup, find_packages

install_requires = [
    "torch>=1.7.1",
    "gym>=0.9.7",
    "numpy>=1.10.4",
    "pillow",
    "filelock",
]

setup(name="rlrl", version="1.0.0", description="Library for RL research", author="hisaki", packages=["rlrl"])
