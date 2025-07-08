from setuptools import setup, find_packages

setup(
    name="bipolar-app",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.2",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0"
    ],
    python_requires=">=3.10",
)