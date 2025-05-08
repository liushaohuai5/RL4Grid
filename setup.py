from setuptools import setup, find_packages

setup(
    name="RL4Grid",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "gym==0.17.1",
        "numpy==1.21.0",
        "pypower",
    ],
    author="Shaohuai Liu",
    author_email="liushaohuai42@gmail.com",
    description="An RL Env for optimal dispatching",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liushaohuai5/RL4Grid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)