run: clean kernel
	g++ \
		-std=c++17 -fPIC -O3 \
		-I . -I ../../include \
		-l OpenCL\
		main.cpp -o run
	./run
kernel:
	rm -f kernel.h
	touch kernel.h
	echo "#ifndef __KERNEL_H__" >> kernel.h 
	echo "#define __KERNEL_H__" >> kernel.h
	xxd -i -n kernel kernel.cl >> kernel.h
	echo "#endif // __KERNEL_H__" >> kernel.h
clean:
	rm -f run kernel.h
