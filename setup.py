from setuptools import setup

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="micromodelsim",
    version="0.0.1",
    description="Microstructural model simulator",
    long_description=long_description,
    url="https://github.com/kerkelae/micromodelsim",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["micromodelsim", "micromodelsim.tests"],
    install_requires=["healpy", "numpy", "scipy"],
    include_package_data=True,
    package_data={"": ["license.txt"]},
)
