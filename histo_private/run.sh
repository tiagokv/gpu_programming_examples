#!/bin/bash

rm log* output_gpu*.ppm

./histo_private -e data/0/output.ppm -i data/0/input.ppm -o output_gpu0.ppm -t image > log0
./histo_private -e data/1/output.ppm -i data/1/input.ppm -o output_gpu1.ppm -t image > log1
./histo_private -e data/2/output.ppm -i data/2/input.ppm -o output_gpu2.ppm -t image > log2
./histo_private -e data/3/output.ppm -i data/3/input.ppm -o output_gpu3.ppm -t image > log3
./histo_private -e data/4/output.ppm -i data/4/input.ppm -o output_gpu4.ppm -t image > log4
./histo_private -e data/5/output.ppm -i data/5/input.ppm -o output_gpu5.ppm -t image > log5
./histo_private -e data/6/output.ppm -i data/6/input.ppm -o output_gpu6.ppm -t image > log6
./histo_private -e data/7/output.ppm -i data/7/input.ppm -o output_gpu7.ppm -t image > log7
./histo_private -e data/8/output.ppm -i data/8/input.ppm -o output_gpu8.ppm -t image > log8
./histo_private -e data/9/output.ppm -i data/9/input.ppm -o output_gpu9.ppm -t image > log9