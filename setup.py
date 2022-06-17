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
    package_dir={'': 'picutils'}, 
    packages=setuptools.find_packages(where='picutils'), 
    url='https://blog.picpic.site', 
    license='MIT', 
    author='PIC', 
    author_email='picpic2019@gmail.com', 
    description='a dibr torch lib', 
    long_description=long_description, 
    long_description_content_type="text/markdown",
    python_requires=">=3.8, <4", 
    install_requires=[
        'numpy>=1.22.3', 
        'opencv-python>=4.5.5.64', 
        'plyfile>=0.7.4', 
        'torch>=1.11.0'
    ]
)