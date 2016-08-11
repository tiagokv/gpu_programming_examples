# Heterogenous Parallel Programming (GPU)

This repo was created for sharing the courseworks from the Heterogenous Parallel Programming that can be found in this [site](http://webgpu.com/), even though the course is deprecated and no longer available in Coursera (I download the videos to my pc).

For the purpose of learning Docker as well, I created a Dockerfile to instantiate a development environment with the prerequisites, which is basically this library in this [repository](https://github.com/abduld/libwb). I use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin to instantiate the GPU dependent containers. You will see within the Dockerfile that is based in the Ubuntu 14.04 and CUDA 7.5, you should change it to what you have installed in your host.

If the name starts with "opencl", means that uses OpenCL library, otherwise assume CUDA. Each folder contains:
 - a Makefile: Oschestrate **nvcc** compiler on how to compile and link the program itself. 
 - run.sh: Run the program against a dataset and store the log trace printed to a log file

I use the following command in docker to start the container:

```shell
nvidia-docker run -v "$(PWD)":/repos -w /repos -it nvidia/cuda:7.5-devel-ubuntu14.04 /bin/bash
```

where PWD should be your directory containing this repository. You could change the **/bin/bash** to execute one of the run.sh.

Have fun! :)