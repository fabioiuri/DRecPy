sudo rm -rf ../build/
sphinx-apidoc -o ./api_docs/ ../../DRecPy
cd ..
sudo make html