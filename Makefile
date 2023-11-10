all: pgm.o hough

hough: houghProyecto.cu pgm.o
	nvcc houghProyecto.cu pgm.o -o hough

pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o pgm.o

clean:
	rm -f hough pgm.o