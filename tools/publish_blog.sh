#!/bin/bash

set -e

DEST=$HOME/workspace/anarthal.github.io/kernel

rm -rf $DEST
mkdir $DEST
cd ~/workspace/blog
bash tools/build.sh -b /kernel -d $DEST
cd $HOME/workspace/anarthal.github.io
git add "kernel/*"
git commit -m "Updated personal blog on $(date '+%Y-%m-%d')"
git push
