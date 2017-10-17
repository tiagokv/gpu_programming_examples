LIBWB_PATH=/repositories/libwb/
OUTPUT=scan_sum

$(OUTPUT): *.cu
	nvcc -g -G -I$(LIBWB_PATH) -std=c++11 $< -lwb -o $@

.PHONY: clean

clean:
	rm -f $(OUTPUT) log