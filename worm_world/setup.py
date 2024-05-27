from setuptools import setup, find_packages

setup(
    name='worms',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gym~=0.26.0',
        'gymnasium~=0.29.1',
        'pygame~=2.1.0',
        'numpy~=1.25.2',
        'torch~=2.0.0'
    ],
    author='Anna',
    author_email='your.email@example.com',
    description='A short description of your package',
)
