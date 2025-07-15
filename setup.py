import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tabular-transformers",
    version="0.1.1",
    author="Shivansh Gupta",
    author_email="duster.amigos05@gmail.com",
    description="Production-ready PyTorch implementations of Transformers for tabular data (numerical/categorical, multi-output, multi-label)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duster-amigos/Tabular-Deep-Learning",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.2"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    include_package_data=True,
    project_urls={
        "Source": "https://github.com/duster-amigos/Tabular-Deep-Learning",
        "Documentation": "https://github.com/duster-amigos/Tabular-Deep-Learning#readme"
    },
) 