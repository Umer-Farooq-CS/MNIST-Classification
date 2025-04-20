# Compiler and flags
CC = gcc
CFLAGS = -O3 -fopenacc -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -fopenacc

# Paths and filenames
SRC_DIR = .
BUILD_DIR = build
PROF_DIR = $(BUILD_DIR)/prof
TIME_FILE = $(BUILD_DIR)/training_time.txt
EXEC = $(BUILD_DIR)/mnist_nn
PROF_EXEC = $(BUILD_DIR)/mnist_nn_prof
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))
PROF_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.prof.o, $(SRCS))
GMON_OUT = $(BUILD_DIR)/gmon.out
NVPROF_LOG = $(BUILD_DIR)/nvprof_log.txt
OUTPUT_FILE = $(BUILD_DIR)/full_output.txt

# Default target
all: $(EXEC)

# Ensure build directories exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(PROF_DIR)

# Compile normal object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile profiling object files
$(BUILD_DIR)/%.prof.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -pg -c $< -o $@

# Link normal executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Link profiling executable
$(PROF_EXEC): $(PROF_OBJS)
	$(CC) $(CFLAGS) -pg -o $@ $^ $(LDFLAGS)

# Run the program, save time to file
run: $(EXEC)
	@echo "Running neural network..."
	@./$(EXEC) | tee $(OUTPUT_FILE)
	@grep "Total training time" $(OUTPUT_FILE) | tee $(TIME_FILE)
	@mv gmon.out $(GMON_OUT) 2>/dev/null || true

# Run profiling for CPU (gprof)
prof-run: $(PROF_EXEC)
	@echo "Running with profiling..."
	@./$(PROF_EXEC)
	@mv gmon.out $(GMON_OUT) 2>/dev/null || true

# Generate profiling report and graph (CPU profiling)
prof: prof-run
	@echo "Generating profiling report..."
	@mkdir -p $(PROF_DIR)
	@gprof $(PROF_EXEC) $(GMON_OUT) > $(PROF_DIR)/gprof_analysis.txt
	@gprof $(PROF_EXEC) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null
	@echo "Profiling results saved to:"
	@echo "  - $(PROF_DIR)/gprof_analysis.txt"
	@echo "  - $(PROF_DIR)/gprof_graph.png"

# Run GPU profiling with nvprof (without OpenACC profiling flags)
gpu-prof-run: $(EXEC)
	@echo "Running GPU profiling with nvprof..."
	@nvprof --output-profile $(NVPROF_LOG) ./$(EXEC)

# Generate GPU profiling report
gpu-prof: gpu-prof-run
	@echo "GPU Profiling results saved to $(NVPROF_LOG)"
	@echo "You can analyze the results with nvprof or visualize it with nvvp."

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean run prof prof-run gpu-prof gpu-prof-run
