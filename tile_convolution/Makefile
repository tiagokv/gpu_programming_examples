LIBWB_PATH=/repositories/libwb/
OUTPUT=tile_convolution

$(OUTPUT): *.cu
	nvcc -g -G -I$(LIBWB_PATH) -std=c++11 $< -lwb -o $@

.PHONY: clean

clean:
	rm -f $(OUTPUT) log