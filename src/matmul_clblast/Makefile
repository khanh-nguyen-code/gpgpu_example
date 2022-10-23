bench: clean
	g++ \
		-std=c++17 -fPIC -O3 \
		-I . -I ../../include \
		-l OpenCL -l clblast -fopenmp\
		bench.cpp -o bench
	./bench
run: clean
	g++ \
		-std=c++17 -fPIC -O3 \
		-I . -I ../../include \
		-l OpenCL -l clblast -fopenmp\
		main.cpp -o run 
	./run
clean:
	rm -f run bench
