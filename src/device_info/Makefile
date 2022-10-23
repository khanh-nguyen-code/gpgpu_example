run: clean 
	g++ \
		-std=c++17 -fPIC -O3\
		-I ../../include \
		-l OpenCL \
		main.cpp -o run
	./run
clean:
	rm -f run
