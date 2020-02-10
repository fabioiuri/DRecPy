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
    version="0.0.1",
    author="Fabio Colaço",
    author_email="fcolaco@lasige.di.fc.ul.pt",
    description="todo", #todo
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", # todo
    packages=find_packages(exclude=['tests*']),
    classifiers=[ ], # todo
    keywords="todo" #todo
)