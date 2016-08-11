LIBWB_PATH=/repositories/libwb/
OUTPUT=vector_addition

$(OUTPUT): *.cu
	nvcc -I$(LIBWB_PATH) -std=c++11 $< -lwb -o $@

.PHONY: clean

clean:
	rm -f $(OUTPUT) log