from setuptools import setup, find_packages

setup(
    name='labicompare',
    version='0.1.0',
    packages=find_packages(where='labicompare'),
    package_dir={'': 'labicompare'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pyyaml',
    ],
)
