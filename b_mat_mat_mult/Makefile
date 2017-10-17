LIBWB_PATH=/repositories/libwb/
OUTPUT=b_mat_mat_mult

$(OUTPUT): *.cu
	nvcc -I$(LIBWB_PATH) -std=c++11 $< -lwb -o $@

.PHONY: clean

clean:
	rm -f $(OUTPUT) log