from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hypster",
    version="0.1.0",
    author="Gilad Rubin",
    author_email="gilad.rubin@gmail.com",
    description="A flexible configuration system for Python projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hypster-dev/hypster",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "streamlit>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "isort",
            "mypy",
            "flake8",
        ],
    },
)