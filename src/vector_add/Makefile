run: clean
	xxd -i kernel.cl > kernel.cl.h
	g++ \
		-std=c++17 -fPIC -O3 \
		-I . -I ../../include \
		-l OpenCL\
		main.cpp -o main.out
	./main.out
clean:
	rm -f main.out
