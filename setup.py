import os
from setuptools import setup, find_packages

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('README.md') as f:
    readme = f.read()

with open('install_requires.txt') as f:
    install_requires = f.read()

setup(
    name='tf_speed_test',
    version='0.0.1',
    packages=find_packages(exclude=('test', 'examples')),
    zip_safe=True,
    include_package_data=False,
    description='Testing all the speed',
    long_description=readme,
    author='Dan Busbridge',
    author_email='dan.busbridge@babylonhealth.com',
    url='https://github.com/dbusbridge/tf_speed_test',
    install_requires=install_requires
)
