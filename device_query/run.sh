#!/bin/bash

if [ -f "log" ]
then
	rm log
fi

./device_query > log