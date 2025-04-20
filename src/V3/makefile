CC = nvcc
FLAGS = -O3 -Xcompiler -Wall,-Wextra,-ffast-math,-funroll-loops -pg
NVFLAGS = -gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86
LDFLAGS = -lcublas -lcurand
EXE = build/nn.exe
PROF_EXE = build/nn_prof.exe
SRC = main.cu neural_net.cu utils.cu mnist.cu
OBJ = $(patsubst %.cu,build/%.o,$(SRC))
PROF_OBJ = $(patsubst %.cu,build/%.prof.o,$(SRC))
PROF_DIR = build/prof
NSIGHT_DIR = nsight_results
GMON_OUT = build/gmon.out
NSIGHT_REPORT = $(NSIGHT_DIR)/cpu_profile_report
TRAINING_TIME_FILE = build/training_time.txt

# Speedup analysis
SPEEDUP_CC = gcc
SPEEDUP_SRC = speedup_analysis.c
SPEEDUP_EXE = build/speedup_analysis

all: $(EXE)

$(EXE): build $(OBJ)
	$(CC) $(FLAGS) $(NVFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS)

$(PROF_EXE): build $(PROF_OBJ)
	$(CC) $(FLAGS) $(NVFLAGS) -o $(PROF_EXE) $(PROF_OBJ) $(LDFLAGS)

build:
	mkdir -p build $(PROF_DIR) $(NSIGHT_DIR)

build/%.o: %.cu | build
	$(CC) $(FLAGS) $(NVFLAGS) -dc $< -o $@

build/%.prof.o: %.cu | build
	$(CC) $(FLAGS) $(NVFLAGS) -dc $< -o $@

run: $(EXE)
	./$(EXE) | tee build/full_output.txt
	grep "Total training time" build/full_output.txt > build/training_time.txt
	mv gmon.out build/ 2>/dev/null || true

prof-run: $(PROF_EXE)
	./$(PROF_EXE) | tee $(TRAINING_TIME_FILE)
	mv gmon.out build/ 2>/dev/null || true

# Nsight profiling step
nsight-profile: $(EXE)
	mkdir -p $(NSIGHT_DIR)
	nsys profile --stats=true --force-overwrite=true --output=$(NSIGHT_REPORT) ./$(EXE)
	@echo "Nsight Systems profiling results saved in $(NSIGHT_REPORT)."

# Analyze Nsight profiling report
nsight-analyze:
	@echo "Analyzing Nsight profiling report..."
	nsys stats $(NSIGHT_REPORT).nsys-rep
	@echo "Summary report generated."
	nsys export --format=csv $(NSIGHT_REPORT).nsys-rep
	@echo "Profiling data exported to CSV format."
	@echo "Opening Nsight Systems GUI..."
	nsight-sys $(NSIGHT_REPORT).nsys-rep
	@echo "Nsight Systems GUI opened with profiling report."

# Profiling with gprof and visualization
prof: prof-run
	@echo "Generating profiling report..."
	gprof $(PROF_EXE) $(GMON_OUT) > $(PROF_DIR)/gprof_analysis.txt
	gprof $(PROF_EXE) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null
	@echo "Profiling results saved to:"
	@echo "  - $(PROF_DIR)/gprof_analysis.txt"
	@echo "  - $(PROF_DIR)/gprof_graph.png"

# Compile and run speedup analysis
$(SPEEDUP_EXE):
	mkdir -p ../build
	$(SPEEDUP_CC) -o $(SPEEDUP_EXE) $(SPEEDUP_SRC)

# Target for speedup analysis
speedup: $(SPEEDUP_EXE)
	@echo "Running speedup analysis..."
	@./$(SPEEDUP_EXE)

clean:
	rm -rf build $(NSIGHT_DIR)

.PHONY: all run prof-run clean prof nsight-profile nsight-analyze speedup