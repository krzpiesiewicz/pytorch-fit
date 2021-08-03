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
    packages=setuptools.find_packages(exclude=['tests', 'examples']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=1.0.5",
        "numpy~=1.19.0",
        "torch>=1.8.0",
        "matplotlib>=3.2.2",
        "IPython>=5.5.0",
        "plotly>=4.5.0",
    ],
    test_requirements=["pytest>=6.2.0"],
    python_requires='>=3.6',
)
