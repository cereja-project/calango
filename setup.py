import setuptools

import magic
from cereja.file import FileIO

long_description = b''.join(FileIO.load('README.md').data).decode()
REQUIRED_PACKAGES = FileIO.load("requirements.txt")

EXCLUDE_FROM_PACKAGES = ()

setuptools.setup(
        name="magic",
        version=magic.__version__,
        author="Joab Leite",
        author_email="jlsn1@ifal.edu.br",
        description="It looks like magic",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/cereja-project/magic",
        packages=setuptools.find_packages(exclude=EXCLUDE_FROM_PACKAGES),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=REQUIRED_PACKAGES
)
