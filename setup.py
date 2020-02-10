"""
Release steps:
- Check if tests are passing
- Update and build new dobs (make clean, make html)
- Add git release tag (git tag v0.0.0, git push --tags)
- Build package (python setup.py sdist bdist_wheel)
- Upload package (python -m twine upload dist/*)
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="DRecPy",
    version="0.0.0",
    author="Fabio Colaço",
    author_email="fabioiuri@live.com",
    description="Description Deep Recommenders with Python: "
                "A Python framework for building Deep Learning based Recommender Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabioiuri/DRecPy",
    packages=find_packages(exclude=['tests*']),
    classifiers=['Intended Audience :: Developers', 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research', 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3', 'Development Status :: 2 - Pre-Alpha',
                 'Topic :: Scientific/Engineering'],
    keywords="recommender, recommendation, system, machine learning, deep learning"
)