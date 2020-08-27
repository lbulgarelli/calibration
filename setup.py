# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='calibration-belt',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',
    description='Assessment of calibration in binomial prediction models.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/lbulgarelli/calibration',

    # Author details
    author='Lucas Bulgarelli',
    author_email='lucas1@mit.edu',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='calibration calibration-belt p-value goodness-of-fit',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'tests'])

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    py_modules=['calibration'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy>=1.19.1', 'tqdm>=4.48.2', 'matplotlib>=3.3.0',
        'scipy>=1.5.2', 'statsmodels>=0.11.1'
    ]
)
