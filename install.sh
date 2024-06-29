#!/bin/bash

mkdir -p third_party
cd third_party && rm -rf eigen-3.4.0
curl -OL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip
rm -f eigen-3.4.0.zip
cd - && pip install -r requirements.txt
pip install .
