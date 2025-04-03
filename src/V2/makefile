CC = nvcc
FLAGS = -O3 -Xcompiler -Wall,-Wextra,-ffast-math,-funroll-loops -pg
NVFLAGS = -gencode arch=compute_61,code=sm_61 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86 \
          --default-stream per-thread

LDFLAGS = -lcublas -lcurand -lcudart
EXE = build/nn.exe
PROF_EXE = build/nn_prof.exe
SRC = main.cu neural_net.cu utils.cu mnist.cu
OBJ = $(patsubst %.cu,build/%.o,$(SRC))
PROF_OBJ = $(patsubst %.cu,build/%.prof.o,$(SRC))
PROF_DIR = build/prof
GMON_OUT = build/gmon.out

all: $(EXE)

$(EXE): build $(OBJ)
	$(CC) $(FLAGS) $(NVFLAGS) -o $(EXE) $(OBJ) $(LDFLAGS)

$(PROF_EXE): build $(PROF_OBJ)
	$(CC) $(FLAGS) $(NVFLAGS) -o $(PROF_EXE) $(PROF_OBJ) $(LDFLAGS)

build:
	mkdir -p build $(PROF_DIR)

build/%.o: %.cu | build
	$(CC) $(FLAGS) $(NVFLAGS) -dc $< -o $@

build/%.prof.o: %.cu | build
	$(CC) $(FLAGS) $(NVFLAGS) -dc $< -o $@

run: $(EXE)
	./$(EXE)
	mv gmon.out $(GMON_OUT) 2>/dev/null || true
	
prof-run: $(PROF_EXE)
	./$(PROF_EXE)
	mv gmon.out $(GMON_OUT) 2>/dev/null || true

clean:
	rm -rf build

prof: prof-run
	@echo "Generating profiling report..."
	gprof $(PROF_EXE) $(GMON_OUT) > $(PROF_DIR)/gprof_analysis.txt
	gprof $(PROF_EXE) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null
	@echo "Profiling results saved to:"
	@echo "  - $(PROF_DIR)/gprof_analysis.txt"
	@echo "  - $(PROF_DIR)/gprof_graph.png"

.PHONY: all run prof-run clean prof