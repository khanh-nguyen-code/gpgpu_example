run: clean
	g++ \
		-std=c++17 -fPIC -O3 \
		-I . -I ../include \
		-l OpenCL -l clblast\
		main.cpp -o main.out
	./main.out
clean:
	rm -f main.out
