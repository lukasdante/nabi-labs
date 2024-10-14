#!/bin/sh
# change directory to parent IsaacLab


cd ${ISAAC_LAB}

mkdir ../arctos/arctos_urdf_description/urdf/$1

# run the converter
./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
  ../arctos/arctos_urdf_description/urdf/$1.urdf \
  ../arctos/arctos_usd/$1/$1.usd \
  --merge-joints \
  --make-instanceable \
  --fix-base \
  --headless