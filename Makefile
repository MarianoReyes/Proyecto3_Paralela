all: pgm.o hough_base hough_const hough_shared


hough_base: houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -ljpeg -o hough_base

hough_const: houghConst.cu pgm.o
	nvcc houghConst.cu pgm.o -ljpeg -o hough_const

hough_shared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -ljpeg -o hough_shared

pgm.o: pgm.cpp
	g++ -c pgm.cpp -o pgm.o

clean:
	rm -f hough pgm.o

run_base:
	./hough_base runway.pgm 4500

run_const:
	./hough_const runway.pgm 4500

run_shared:
	./hough_shared runway.pgm 4500