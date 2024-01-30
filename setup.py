import os
from setuptools import setup, find_packages, Command, Extension


__version__ = None
exec(open('scrna_tools/version.py').read())


class CleanCommand(Command):
    """
    Custom clean command to tidy up the project root.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./htmlcov')


setup(
    name='scrna_tools',
    version=__version__,
    description='Consolidated scrna-seq analyses.',
    setup_requires=[
        'setuptools>=18.0',
        'cython'
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.0',
        'scipy',
        'matplotlib>=3.1.0',
        'pandas>=0.15.0',
        'anndata',
        'scanpy',
        'requests',
        'gseapy',
        'h5py',
        'adjustText',
        'loompy',
        'future',
        'leidenalg',
        'harmonypy',
        'scrublet',
        'cmocean',
        'patsy',
        #'scvi_tools',
        'pyarrow',
        'regex',
        'anndataview',
        'pydeseq2',
        'textalloc'
    ],
    cmdclass={
        'clean': CleanCommand
    },
    scripts=['bin/rhapsody-extract-barcode', 'bin/rhapsody-demultiplex'],
)
