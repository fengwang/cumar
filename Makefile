CXX           = clang++
CXXFLAGS        = -std=c++1z -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command
INCPATH       = -Iinclude -I/Developer/NVIDIA/CUDA-8.0/include -I/opt/cuda/include -I/usr/local/cuda-7.0/include
LINK          = $(CXX)
LFLAGS        = -lc++ -lc++abi -O3 -lcudart -lnvrtc -L/Developer/NVIDIA/CUDA-8.0/lib -framework CUDA      ## mac config
#LFLAGS        = -lc++ -lc++abi -O3 -lcudart -lnvrtc -lcuda -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 ## linux config
DEL_FILE      = rm -f

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib

all: map reduce

clean:
	rm -rf $(OBJECTS_DIR)/*
	rm -rf $(BIN_DIR)/*
	rm -rf $(LIB_DIR)/*
	rm -rf ./ptx/*

cumar.o: src/cumar.cc
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/cumar.o src/cumar.cc

map: test/map.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/map.o test/map.cc -Wunreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/map ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/map.o

reduce: test/reduce.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/reduce.o test/reduce.cc -Wno-unreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/reduce ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/reduce.o

