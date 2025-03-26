CC = gcc
CFLAGS = -Wall -O2 -pg
EXE = build/nn.exe
SRC = nn.c
OBJ = build/nn.o
GMON_OUT = build/gmon.out

all: $(EXE) run

$(EXE): build $(OBJ)
	$(CC) $(CFLAGS) -o $(EXE) $(OBJ) -lm

build:
	mkdir -p build

build/nn.o: $(SRC) | build
	$(CC) $(CFLAGS) -c $(SRC) -o $(OBJ)

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