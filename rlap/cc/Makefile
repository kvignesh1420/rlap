CC=g++
CXXFLAGS=-I ../eigen-3.4.0 -std=c++11 -march=native -mfpmath=sse -O3 -DEIGEN_MPL2_ONLY

solver: solver.cc  reader.cc reader.h factorizers.cc factorizers.h samplers.cc samplers.h cg.cc cg.h tracer.h types.h
	$(CC) $(CXXFLAGS) -o solver solver.cc reader.cc factorizers.cc samplers.cc cg.cc

.PHONY: clean

clean:
	rm -f solver *~