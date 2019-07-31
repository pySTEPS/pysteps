# Build documentation from scratch.

rm -r source/generated &> /dev/null
rm -r source/auto_examples &> /dev/null

make clean

make html
