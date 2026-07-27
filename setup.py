from setuptools import setup, find_packages

setup(
    name="numerical_lib",
    version="1.0",
    author="Dmitriy Nikitenko",
    author_email="ddimyc34@gmail.com",
    description="A Python library for numerical methods and solving ODEs.",
    
    # The source code is in the src folder
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
)