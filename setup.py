import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cooke_book',
    version='0.0.1',
    author='Jonathan Cooke',
    author_email='cooke.jon.mba@gmail.com',
    description='Data Science Functions for personal use',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/byemoney/ds_toolkit',
    license='MIT',
    packages=['cooke_book],
    install_requires=['requests'],
)