#!/bin/bash

if [ -f "log" ]
then
	rm log
fi

if [ -f "output_gpu.raw" ]
then
	rm output_gpu.raw
fi

./vector_addition -e ../data/1/output.raw -i ../data/1/input0.raw,../data/1/input1.raw -o output_gpu.raw -t vector > log