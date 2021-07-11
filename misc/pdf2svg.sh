#!/bin/bash -l

# Requires an installation of PDF2SVG (http://cityinthesky.co.uk/opensource/pdf2svg/)

for i in ../results/figures/*.pdf; do
   pdf2svg $i ${i%.pdf*}.svg
done