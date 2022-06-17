from setuptools import setup
import setuptools
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

with open(os.path.join(here, 'README.md'), 'r') as mdFile:
    long_description = mdFile.read()

setup(
    name='picutils', 
    version='0.0.1', 
    packages=setuptools.find_packages(), 
    url='https://blog.picpic.site', 
    license='MIT', 
    author='PIC', 
    author_email='picpic2019@gmail.com', 
    description='a dibr torch lib', 
    long_description=long_description, 
    long_description_content_type="text/markdown",
    python_requires=">=3.7, <4"
)