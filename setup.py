from setuptools import setup, find_packages

setup(
    name='multilayerGM',
    description='Generate multilayer networks with mesoscale structure',
    url='https://github.com/MultilayerGM/MultilayerGM-py',
    version='0.0.0',
    author='Lucas G. S. Jeub',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
