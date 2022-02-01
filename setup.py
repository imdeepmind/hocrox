"""Setup for PyPI release."""
import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

description = "Hocrox is an image preprocessing and augmentation library. It provides a \
Keras like simple interface to make preprocessing and augmentation pipelines."

setuptools.setup(
    name="Hocrox",  # This is the name of the package
    version="0.3.0",  # The initial release version
    author="Abhishek Chatterjee",  # Full name of the author
    author_email="Abhishek.Chatterjee97@protonmail.com",
    description=description,
    long_description=long_description,  # Long description read from the the readme file
    long_description_content_type="text/markdown",
    classifiers=[
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],  # Information to filter the project on PyPi website
    python_requires=">=3.6",  # Minimum version requirement of the package
    packages=find_packages("."),  # Name of the python package
    install_requires=["opencv-python", "tqdm", "prettytable"],  # Install other dependencies if any
    project_urls={
        "Bug Tracker": "https://github.com/imdeepmind/hocrox/issues",
        "Documentation": "https://hocrox.imdeepmind.com/",
        "Source Code": "https://github.com/imdeepmind/hocrox",
    },
)
