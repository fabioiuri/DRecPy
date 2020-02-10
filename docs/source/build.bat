cd ..
rmdir /S build
mkdir build
sphinx-apidoc -o build\doctrees\api_docs\ ..\DRecPy\
make html
cd source
