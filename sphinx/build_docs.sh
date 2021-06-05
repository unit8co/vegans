rm -rf docs/*
cd ./sphinx/source
rm -f README.md
ln -s ../../README.md
rm -f copyright.rst
ln -s ../../LICENSE copyright.rst
cd ../
sphinx-apidoc -f -o ../sphinx/source/ ../vegans
make clean html
cp -r ./build/html/* ../docs/
cd ../
echo -e "---\npermalink: /index.html\n---" > docs/readme.md
touch docs/.nojekyll