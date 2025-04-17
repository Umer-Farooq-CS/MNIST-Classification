# Compiler and flags
CC = gcc
CFLAGS = -O3 -fopenacc -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -fopenacc

# Source files
SRCS = main.c neural_net.c mnist.c utils.c
OBJS = $(SRCS:.c=.o)
EXEC = mnist_nn

# Default build target
all: $(EXEC)

# Link object files into executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program after building
run: all
	./$(EXEC)

# Clean up build files
clean:
	rm -f $(OBJS) $(EXEC)

.PHONY: all clean run
