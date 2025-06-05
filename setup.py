from setuptools import setup

setup(
    name='richmol',
    version='0.1',
    packages=['richmol'],
    package_data={'richmol': ['expokit*.so']},
    include_package_data=True,
)