CC = gcc
CFLAGS = -Wall -O2 -pg -Isrc
EXE = build/nn.exe
SRC = main.c neural_net.c utils.c mnist.c
OBJ = $(patsubst %.c,build/%.o,$(SRC))
GMON_OUT = build/gmon.out

all: $(EXE) run

$(EXE): build $(OBJ)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ) -lm

build:
	mkdir -p build

build/%.o: %.c | build
	$(CC) $(CFLAGS) -c $< -o $@

run: $(EXE)
	./$(EXE)
	mv gmon.out $(GMON_OUT) 2>/dev/null || true

clean:
	rm -rf build prof_analysis

# Profiling with gprof and visualization
prof: run
	mkdir -p prof_analysis
	gprof $(EXE) $(GMON_OUT) > prof_analysis/gprof_analysis.txt
	gprof $(EXE) $(GMON_OUT) -q | gprof2dot -n 0 -e 0 | dot -Tpng -o prof_analysis/gprof_graph.png 2>/dev/null
	@echo "Profiling analysis saved in prof_analysis/gprof_analysis.txt and call graph saved in prof_analysis/gprof_graph.png"