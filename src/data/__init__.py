"""Setup script for housing-mlops project."""

from setuptools import setup, find_packages

setup(
    name="housing-mlops",
    version="1.0.0",
    description="California Housing Price Prediction MLOps Pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
)