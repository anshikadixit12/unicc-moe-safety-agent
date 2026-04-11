from setuptools import setup, find_packages

setup(
    name="unicc-moe-safety-agent",
    version="1.0.0",
    description="UNICC Mixture of Experts AI Safety Testing Platform",
    author="Anshika Dixit",
    author_email="ad7610@nyu.edu",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
)
