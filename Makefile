all: pgm.o hough_base hough_const


hough_base: houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -ljpeg -o hough_base

# hough_const: houghConst.cu pgm.o
# 	nvcc houghConst.cu pgm.o -ljpeg -o hough_const

pgm.o: pgm.cpp
	g++ -c pgm.cpp -o pgm.o

clean:
	rm -f hough pgm.o

run_base:
	./hough_base runway.pgm 4500

# run_const:
# 	./hough_const 4000