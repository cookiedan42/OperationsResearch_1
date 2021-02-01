## EXAMPLE SETUP

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OR_one",
    version="0.0.1",
    author="Daniel Tan",
    author_email="cookiedan42@gmail.com",
    description="helper functions for IE2111 problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cookiedan42/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pandas',"matplotlib","networkx"]
)
