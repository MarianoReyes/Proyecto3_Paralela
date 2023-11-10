all: pgm.o hough_base hough_const


hough_base: houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough_base

hough_const: houghConst.cu pgm.o
	nvcc houghConst.cu pgm.o -o hough_const

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o

clean:
	rm -f hough pgm.o

run_base:
	./hough_base

run_const:
	./hough_const