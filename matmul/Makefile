run: clean 
	g++ -std=c++17 -fPIC -O3 -I../include -lOpenCL -lopenblas main.cpp -o main.out
	./main.out
clean:
	rm -f main.out
