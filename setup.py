from setuptools import find_packages, setup
from setuptools_rust import RustExtension

with open('README.md', 'r') as inf:
    long_description = inf.read()

setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='woods',
    version='0.1.0',
    author='Andrei V. Konstantinov',
    author_email='andrue.konst@gmail.com',
    description='Decision Trees Ensembles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/andruekonst/Woods',
    # packages=['woods'],
    rust_extensions=[RustExtension(
        'woods',
        'Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    # packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
