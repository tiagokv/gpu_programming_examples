#!/bin/bash

rm log* output_gpu*.raw

./scan_sum -e data/0/output.raw -i data/0/input.raw -o output_gpu0.raw -t vector > log0
./scan_sum -e data/1/output.raw -i data/1/input.raw -o output_gpu1.raw -t vector > log1
./scan_sum -e data/2/output.raw -i data/2/input.raw -o output_gpu2.raw -t vector > log2
./scan_sum -e data/3/output.raw -i data/3/input.raw -o output_gpu3.raw -t vector > log3
./scan_sum -e data/4/output.raw -i data/4/input.raw -o output_gpu4.raw -t vector > log4
./scan_sum -e data/5/output.raw -i data/5/input.raw -o output_gpu5.raw -t vector > log5
./scan_sum -e data/6/output.raw -i data/6/input.raw -o output_gpu6.raw -t vector > log6
./scan_sum -e data/7/output.raw -i data/7/input.raw -o output_gpu7.raw -t vector > log7
./scan_sum -e data/8/output.raw -i data/8/input.raw -o output_gpu8.raw -t vector > log8
./scan_sum -e data/9/output.raw -i data/9/input.raw -o output_gpu9.raw -t vector > log9