# Image to execute 
FROM nvidia/cuda:7.5-devel-ubuntu14.04
RUN apt-get update && apt-get install -y \
	git \
	nano \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN mkdir repositories \
	&& cd repositories \
	&& git clone https://github.com/abduld/libwb.git \
	&& cd libwb \
	&& make \
	&& cp lib/libwb.so /usr/lib \
	&& ldconfig