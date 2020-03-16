from setuptools import setup

setup(
    name='LycorisAD',
    version='1.3.4',
    description="An elegant outlier detection algorithm framework based on AutoEncoder.",
    author="RootHarold",
    author_email="rootharold@163.com",
    url="https://github.com/RootHarold/LycorisAD",
    py_modules=['LycorisAD'],
    zip_safe=False,
    install_requires=['LycorisNet>=2.5', 'deap>=1.3', 'scipy>=1.3', 'numpy>=1.18']
)
