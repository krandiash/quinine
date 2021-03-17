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


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

setup(
    name="quinine",
    version="0.2.3",
    author="Karan Goel",
    author_email="kgoel93@gmail.com",
    license="MIT",
    description="quinine is a library for configuring machine learning projects.",
    keywords="configuration yaml machine learning ml ai nlp cv vision deep learning",
    # url="http://packages.python.org/an_example_pypi_project",
    packages=['quinine', 'quinine.common'],
    # long_description=read('README.md'),
    install_requires=install_requires,
)
