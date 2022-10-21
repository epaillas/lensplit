"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

setup(
    name='lensplit',
    version='0.0.1',
    description='',
    url='https://github.com/epaillas/lensplit', 
    author='Enrique Paillas',
        author_email='enrique.paillas@uwaterloo.ca',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6, <4',
    install_requires=[
        'numpy',
        'pypower'
    ],
)
