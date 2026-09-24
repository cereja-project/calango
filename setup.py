import ast
from pathlib import Path

import setuptools

ROOT = Path(__file__).resolve().parent
metadata = ast.parse((ROOT / 'calango' / '_version.py').read_text(encoding='utf-8'))
version = next(ast.literal_eval(node.value) for node in metadata.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets))
long_description = (ROOT / 'README.md').read_text(encoding='utf-8')
REQUIRED_PACKAGES = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()

setuptools.setup(
        name="calango",
        version=version,
        author="Joab Leite",
        author_email="jlsn1@ifal.edu.br",
        description="It looks like calango",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/cereja-project/calango",
        packages=setuptools.find_packages(include=['calango', 'calango.*']),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: 3.14",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.11',
        install_requires=REQUIRED_PACKAGES,
        entry_points={'console_scripts': ['calango-recorder=calango.gui:main']},
)
