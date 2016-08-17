#!/bin/bash

rm log* output_gpu*.raw

./b_mat_mat_mult -e data/0/output.raw -i data/0/input0.raw,data/0/input1.raw -o output_gpu0.raw -t vector > log0
./b_mat_mat_mult -e data/1/output.raw -i data/1/input0.raw,data/1/input1.raw -o output_gpu1.raw -t vector > log1
./b_mat_mat_mult -e data/2/output.raw -i data/2/input0.raw,data/2/input1.raw -o output_gpu2.raw -t vector > log2
./b_mat_mat_mult -e data/3/output.raw -i data/3/input0.raw,data/3/input1.raw -o output_gpu3.raw -t vector > log3
./b_mat_mat_mult -e data/4/output.raw -i data/4/input0.raw,data/4/input1.raw -o output_gpu4.raw -t vector > log4
./b_mat_mat_mult -e data/5/output.raw -i data/5/input0.raw,data/5/input1.raw -o output_gpu5.raw -t vector > log5
./b_mat_mat_mult -e data/6/output.raw -i data/6/input0.raw,data/6/input1.raw -o output_gpu6.raw -t vector > log6
./b_mat_mat_mult -e data/7/output.raw -i data/7/input0.raw,data/7/input1.raw -o output_gpu7.raw -t vector > log7
./b_mat_mat_mult -e data/8/output.raw -i data/8/input0.raw,data/8/input1.raw -o output_gpu8.raw -t vector > log8
./b_mat_mat_mult -e data/9/output.raw -i data/9/input0.raw,data/9/input1.raw -o output_gpu9.raw -t vector > log9
