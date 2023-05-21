from setuptools import setup, find_packages

setup(
    name='cooke_book',
    version='1.0.0',
    author='Jonathan Cooke',
    author_email='cooke.jon.mba@gmail.com',
    description='A Python package for data science utilities',
    long_description='A Python package that provides various functions and utilities for data science projects.',
    url='https://github.com/byemoney/cooke_book',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'category_encoders',
        # Add any other dependencies required by your package
    ],
)
