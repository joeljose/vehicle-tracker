DS_ROOT := /opt/nvidia/deepstream/deepstream
CUDA_INC := /usr/local/cuda-12.8/targets/x86_64-linux/include
INCLUDES := -I$(DS_ROOT)/sources/includes -I$(CUDA_INC)
CXXFLAGS := -shared -fPIC -O2 -std=c++17

libnvds_parse_yolov8.so: nvds_parse_yolov8.cpp
	c++ $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f libnvds_parse_yolov8.so
