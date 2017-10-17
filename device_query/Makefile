LIBWB_PATH=/repositories/libwb/
OUTPUT=device_query

device_query: *.cu
	nvcc -I$(LIBWB_PATH) -std=c++11 $< -lwb -o $@

.PHONY: clean

clean:
	rm -f device_query log