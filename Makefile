# declaration
CC = cc
CFLAGS = -Wall -g -std=c++17 -fPIC -I . -I include -L lib `pkg-config --cflags clblast`
LDFLAGS = -l stdc++ `pkg-config --libs clblast`
LIBRARY = OpenclExample

# generated
LIBRARY_OUT = lib/lib$(LIBRARY).so

UNAME = $(shell uname)
ifeq ($(UNAME), Linux)
LDFLAGS += -l OpenCL -fopenmp -lm
endif
ifeq ($(UNAME), Darwin)
CFLAGS += -I /opt/homebrew/include/ -I/opt/homebrew/opt/libomp/include -L /opt/homebrew/lib -L/opt/homebrew/opt/libomp/lib
LDFLAGS += -framework OpenCL -l omp
endif

TARGET_SRC = $(wildcard *.cpp)
TARGET_OUT = $(patsubst %.cpp,%.out,$(TARGET_SRC))

OBJECT_SRC = $(wildcard src/*.cpp)
OBJECT_OUT = $(patsubst src/%.cpp,obj/%.o,$(OBJECT_SRC))

KERNEL_SRC = $(wildcard kernel/*.cl)
KERNEL_OUT = $(patsubst kernel/%.cl,kernel/%.cl.h,$(KERNEL_SRC))

.PHONY: target directory kernel library clean

# target
target: directory kernel library $(TARGET_OUT)
	@echo "TARGET DONE"	

%.out: %.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -l $(LIBRARY)

# directory
directory:
	mkdir -p obj lib
	@echo "DIRECTORY DONE"

# kernel
kernel: $(KERNEL_OUT)
	@echo "KERNEL DONE"

kernel/%.cl.h: kernel/%.cl
	cat /dev/null > $@
	echo "#ifndef __$(patsubst kernel/%.cl,%,$<)__" >> $@
	echo "#define __$(patsubst kernel/%.cl,%,$<)__" >> $@
	xxd -i $< >> $@
	echo "#endif//__$(patsubst kernel/%.cl,%,$<)__" >> $@

# library
library: $(LIBRARY_OUT)
	@echo "LIBRARY DONE"

$(LIBRARY_OUT): $(OBJECT_OUT)
	$(CC) $(CFLAGS) -shared -o $(LIBRARY_OUT) $(OBJECT_OUT) $(LDFLAGS)

obj/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<


# clean
clean:
	rm -rf $(KERNEL_OUT) $(LIBRARY_OUT) $(TARGET_OUT) obj lib *.dSYM
