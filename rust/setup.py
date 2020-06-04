from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='woods',
    version='0.1.0',
    description='Decision Trees Ensembles',
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
)