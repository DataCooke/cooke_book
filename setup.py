import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cooke_book',
    version='0.0.1',
    author='Jonathan Cooke',
    author_email='cooke.jon.mba@gmail.com',
    description='Data Science functions organized in books. Each user should have one book with thier functions to share',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/byemoney/cooke_book',
    license='MIT',
    packages=['cooke_book],
    install_requires=['requests'],
)