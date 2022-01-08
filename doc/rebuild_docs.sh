# Build documentation from scratch.

rm -r source/generated &> /dev/null
rm -r source/examples_gallery &> /dev/null

make clean

make html
