all:
	nvcc -g -c -o vec_add_device.o vec_add_device.cu
	g++ -g -o main main.cpp *.o -I /opt/cuda/include/ -L /opt/cuda/lib64 -l cudart

clean:
	rm -f main *.o