#!/bin/bash -l

# Requires an installation of Poppler (https://poppler.freedesktop.org/)

for i in *.pdf; do
   pdftoppm -png -singlefile -scale-to-x 4000 -scale-to-y -1 $i ${i%.pdf*}
done