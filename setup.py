"""Setup script for matrix-webcam"""

import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).resolve().parent

# The text of the README file is used as a description
README = (HERE / "README.md").read_text()

LICENSE = (HERE / "LICENSE").read_text()

# This call to setup() does all the work
setup(
    name="matrix-webcam",
    version="0.1.0",
    description="Displays your webcam video feed in the console.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/joschuck/matrix-webcam",
    author="Johannes Schuck",
    author_email="jojoschuck@gmail.com",
    license=LICENSE,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["matrix_webcam"],
    include_package_data=True,
    install_requires=["opencv-python"],
    extras_require={
        "dev": ["pre-commit", "pylint", "black", "mypy"],
    },
    keywords="matrix webcam opencv",
    entry_points={"console_scripts": ["matrix-webcam=matrix_webcam.__main__:main"]},
)
