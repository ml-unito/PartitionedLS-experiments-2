#!/bin/sh

curl -o Dataset.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip

unzip Dataset.zip
cat Dataset/Training/Features_Variant_1.csv Dataset/Testing/Features_TestSet.csv > dataset_orig.csv
