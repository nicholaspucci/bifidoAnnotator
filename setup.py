from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bifidoannotator",
    version="1.0.1",
    author="Nicholas Pucci & Daniel R. Mende",
    author_email="n.pucci@amsterdamumc.nl",
    description="Fine-grained annotation of bifidobacterial enzymes for human milk glycan utilization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicholaspucci/bifidoAnnotator",
    packages=find_packages(),
    package_data={
        'bifidoannotator': [
            'database/mapping_file.tsv',
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",  # Changed from >=3.8
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "bifidoAnnotator=bifidoannotator.cli:main",
        ],
    },
)
