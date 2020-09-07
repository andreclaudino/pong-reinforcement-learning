from setuptools import setup, find_packages

setup(
    name='pong-service',
    version='1.0.0',
    packages=find_packages(),
    url='',
    license='',
    author='Time IA-FRONT',
    author_email='',
    description='',
    install_requires=[
        "flask==1.1.2",
        "click==7.1.2",
        "tensorflow==2.2.0",
        "tf-agents==0.3.0",
        "tensorflow-probability==0.9.0",
    ]
)
