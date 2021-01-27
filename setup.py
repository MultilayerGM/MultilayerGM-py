from setuptools import setup, find_packages

setup(
    name='multilayerGM',
    description='A Python framework for generating multilayer networks with planted mesoscale structure.',
    url='https://github.com/MultilayerGM/MultilayerGM-py',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='Lucas G. S. Jeub',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'nxmultilayer @ git+https://github.com/LJeub/nxMultilayerNet.git#egg=nxmultilayer'
    ],
)
