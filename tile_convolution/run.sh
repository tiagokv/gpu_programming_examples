#!/bin/bash

rm log* output_gpu*.ppm

./tile_convolution -e data/0/output.ppm -i data/0/input0.ppm,data/0/input1.csv -o output_gpu0.ppm -t image > log0
./tile_convolution -e data/1/output.ppm -i data/1/input0.ppm,data/1/input1.csv -o output_gpu1.ppm -t image > log1
./tile_convolution -e data/2/output.ppm -i data/2/input0.ppm,data/2/input1.csv -o output_gpu2.ppm -t image > log2
./tile_convolution -e data/3/output.ppm -i data/3/input0.ppm,data/3/input1.csv -o output_gpu3.ppm -t image > log3
./tile_convolution -e data/4/output.ppm -i data/4/input0.ppm,data/4/input1.csv -o output_gpu4.ppm -t image > log4
./tile_convolution -e data/5/output.ppm -i data/5/input0.ppm,data/5/input1.csv -o output_gpu5.ppm -t image > log5
./tile_convolution -e data/6/output.ppm -i data/6/input0.ppm,data/6/input1.csv -o output_gpu6.ppm -t image > log6
./tile_convolution -e data/7/output.ppm -i data/7/input0.ppm,data/7/input1.csv -o output_gpu7.ppm -t image > log7
./tile_convolution -e data/8/output.ppm -i data/8/input0.ppm,data/8/input1.csv -o output_gpu8.ppm -t image > log8
./tile_convolution -e data/9/output.ppm -i data/9/input0.ppm,data/9/input1.csv -o output_gpu9.ppm -t image > log9
