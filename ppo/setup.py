from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['numpy', 'gym', 'tensorflow-gpu']
setup(name='ppo', version='0.1',
    install_requires=REQUIRED_PACKAGES, include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('ppo')],
    description='Proximal Policy Optimization')