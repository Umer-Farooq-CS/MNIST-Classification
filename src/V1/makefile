# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2 -pg -Isrc
LDFLAGS = -lm

# Paths and filenames
SRC_DIR = .
BUILD_DIR = build
PROF_DIR = $(BUILD_DIR)/prof
EXE = $(BUILD_DIR)/nn.exe
GMON_OUT = $(BUILD_DIR)/gmon.out
TIME_FILE = $(BUILD_DIR)/training_time.txt
OUTPUT_FILE = $(BUILD_DIR)/full_output.txt
SRC = main.c neural_net.c utils.c mnist.c
OBJ = $(patsubst %.c,$(BUILD_DIR)/%.o,$(SRC))

# Default target
all: $(EXE) run

# Ensure build directories exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(PROF_DIR)

# Compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link the object files into the executable
$(EXE): $(OBJ)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS)

# Run the program, save time to file
run: $(EXE)
	@echo "Running neural network..."
	@./$(EXE) | tee $(OUTPUT_FILE)
	@grep "Total training time" $(OUTPUT_FILE) | tee $(TIME_FILE)
	@mv gmon.out $(GMON_OUT) 2>/dev/null || true


# Profiling and moving the gmon output
prof-run: $(EXE)
	@echo "Running with profiling..."
	@./$(EXE)
	@mv gmon.out $(GMON_OUT) 2>/dev/null || true

# Profiling with gprof and visualization
prof: prof-run
	@echo "Generating profiling report..."
	@mkdir -p $(PROF_DIR)
	@gprof $(EXE) $(GMON_OUT) > $(PROF_DIR)/gprof_analysis.txt
	@gprof $(EXE) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null
	@echo "Profiling results saved to:"
	@echo "  - $(PROF_DIR)/gprof_analysis.txt"
	@echo "  - $(PROF_DIR)/gprof_graph.png"

# Clean build and profiling artifacts
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean run prof prof-run
