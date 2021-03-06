CXX           = clang++
CXXFLAGS        = -std=c++17 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef -DDEBUG
INCPATH       = -Iinclude -I/Developer/NVIDIA/CUDA-9.0/include -I/opt/cuda/include -I/usr/local/cuda-7.0/include
LINK          = $(CXX)
#LFLAGS        = -lc++ -lc++abi -O3 -lcudart -lnvrtc -L/Developer/NVIDIA/CUDA-9.0/lib -framework CUDA      ## mac config
LFLAGS        = -lc++ -lc++abi -O3 -lcudart -lnvrtc -lcuda -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 ## linux config
DEL_FILE      = rm -f

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib

all: map reduce

clean:
	rm -rf $(OBJECTS_DIR)/*.o
	rm -rf $(BIN_DIR)/*

cumar.o: src/cumar.cc
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/cumar.o src/cumar.cc

map: test/map.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/map.o test/map.cc -Wunreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/map ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/map.o

reduce: test/reduce.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/reduce.o test/reduce.cc -Wno-unreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/reduce ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/reduce.o

map_1st: test/map_1st.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/map_1st.o test/map_1st.cc -Wunreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/map_1st ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/map_1st.o

reduce_1st: test/reduce_1st.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/reduce_1st.o test/reduce_1st.cc -Wunreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/reduce_1st ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/reduce_1st.o

generate_ptx: test/generate_ptx.cc cumar.o
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/generate_ptx.o test/generate_ptx.cc -Wunreachable-code
	$(LINK) $(LFLAGS) $(LFLAGS) -o $(BIN_DIR)/generate_ptx ${OBJECTS_DIR}/cumar.o $(OBJECTS_DIR)/generate_ptx.o

