# Compiler and flags
CC = gcc
CFLAGS = -O3 -fopenacc -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -fopenacc

# Source files and output folders
SRCS = main.c neural_net.c mnist.c utils.c
OBJS = $(patsubst %.c, build/%.o, $(SRCS))
EXEC = build/mnist_nn

# Default target
all: build $(EXEC)

# Create build directory if not exists
build:
	mkdir -p build

# Link object files into the executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files into object files in the build folder
build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: all
	./$(EXEC)

# Clean up build files
clean:
	rm -rf build

.PHONY: all clean run build
