run: clean 
	g++ -std=c++17 -fPIC -O3 -I../include -l OpenCL -fopenmp main.cpp -o main.out
	./main.out
clean:
	rm -f main.out
