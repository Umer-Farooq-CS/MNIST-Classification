# === Compiler Settings ===
NVCC = nvcc
NVC = nvc

# === Flags ===
CFLAGS   = -acc -Minfo=accel -fast -pg
NVFLAGS  = -O3 -pg -Xcompiler -Wall,-Wextra,-ffast-math,-funroll-loops \
           -gencode arch=compute_61,code=sm_61 \
           -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86

LDFLAGS  = -lcublas -lcurand

# === Source Files ===
CU_SRC  = main.cu mnist.cu
C_SRC   = neural_net.c utils.c

CU_OBJ  = $(CU_SRC:%.cu=build/%.o)
C_OBJ   = $(C_SRC:%.c=build/%.o)

CU_PROF_OBJ = $(CU_SRC:%.cu=build/%.prof.o)
C_PROF_OBJ  = $(C_SRC:%.c=build/%.prof.o)

# === Executables ===
EXE       = build/nn.exe
PROF_EXE  = build/nn_prof.exe
PROF_DIR  = build/prof
GMON_OUT  = build/gmon.out

# === Targets ===
.PHONY: all run clean prof-run prof

# Default build
all: $(EXE)

# Normal executable
$(EXE): build $(CU_OBJ) $(C_OBJ)
	$(NVCC) $(NVFLAGS) -o $@ $(CU_OBJ) $(C_OBJ) $(LDFLAGS)

# Profiling build
$(PROF_EXE): build $(CU_PROF_OBJ) $(C_PROF_OBJ)
	$(NVCC) $(NVFLAGS) -o $@ $(CU_PROF_OBJ) $(C_PROF_OBJ) $(LDFLAGS)

# Ensure build directories exist
build:
	@mkdir -p build $(PROF_DIR)

# === Compile Rules ===

# CUDA Object Files
build/%.o: %.cu | build
	$(NVCC) $(NVFLAGS) -dc $< -o $@

build/%.prof.o: %.cu | build
	$(NVCC) $(NVFLAGS) -dc $< -o $@

# OpenACC C Object Files
build/%.o: %.c | build
	$(NVC) $(CFLAGS) -c $< -o $@

build/%.prof.o: %.c | build
	$(NVC) $(CFLAGS) -c $< -o $@

# === Run Commands ===
run: $(EXE)
	./$(EXE)
	@mv -f gmon.out $(GMON_OUT) 2>/dev/null || true

prof-run: $(PROF_EXE)
	./$(PROF_EXE)
	@mv -f gmon.out $(GMON_OUT) 2>/dev/null || true

# === Profiling Report ===
prof: prof-run
	@echo "Generating profiling report..."
	@gprof $(PROF_EXE) $(GMON_OUT) > $(PROF_DIR)/gprof_analysis.txt
	@gprof $(PROF_EXE) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o $(PROF_DIR)/gprof_graph.png 2>/dev/null
	@echo "Profiling results saved to:"
	@echo "  - $(PROF_DIR)/gprof_analysis.txt"
	@echo "  - $(PROF_DIR)/gprof_graph.png"

# === Clean Build ===
clean:
	rm -rf build
