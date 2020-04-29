"""
Setup script for the Quinine library.
"""
import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Quinine",
    version="0.1dev",
    author="Karan Goel",
    author_email="kgoel93@gmail.com",
    license="BSD",
    description="Quinine is a library for configuring machine learning projects.",
    keywords="configuration yaml machine learning ml ai nlp cv vision deep learning",
    # url="http://packages.python.org/an_example_pypi_project",
    packages=find_packages(),
    long_description=read('README.md'),
)
