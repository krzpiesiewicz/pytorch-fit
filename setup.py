import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-fit",
    version="0.0.1",
    author="Krzysztof Piesiewicz",
    author_email="krz.piesiewicz@gmail.com",
    description="A package consisting of useful tools for automated fitting and evaluating pytorch models. It supports stopping conditions based on metrics, training history visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krzpiesiewicz/pytorch-fit",
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=1.2.0",
        "numpy>=1.20.0",
        "torch>=1.8.0",
        "matplotlib>=3.3.0",
        "IPython>=7.19.0",
    ],
    test_requirements=[],
    python_requires='>=3.6',
)
