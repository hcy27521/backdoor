from setuptools import setup, find_packages

setup(
    name="backdoor",
    version="0.0.1",
    author="Mikel Bober-Irizar",
    packages=find_packages(include=["backdoor", "backdoor.*"]),
    python_requires=">=3.8"
)

