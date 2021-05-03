rm -rf docs/*
cd ./sphinx/source
ln -s ../../README.md
ln -s ../../LICENSE copyright.rst
cd ../../
make --directory ./sphinx clean html
cp -r sphinx/build/html/* docs/
echo -e "---\npermalink: /index.html\n---" > docs/readme.md
touch docs/.nojekyll