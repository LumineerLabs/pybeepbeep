from setuptools import setup, find_packages
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pybeepbeep',
    version_format='{tag}',
    author="Lane Haury",
    author_email="lane@lumineerlabs.com",
    description="pybeepbeep is a python implementation of the BeepBeep cooperative ranging algorithm.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/LumineerLabs/pybeepbeep",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        'numpy',
        'scipy',
        'librosa',
        'pytest-cov',
        'setuptools-git-version',
    ],
)
