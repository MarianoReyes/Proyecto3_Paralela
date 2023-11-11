/*
 ============================================================================
 Author        : Juan Angel Carrera, Juan Carlos Bajan, Jose Mariano Reyes
 Version       : 2.0
 Last modified : November 2023
 License       : Released under the GNU GPL 3.0
 Description   : redo de hough base
 To build use  : make
 To run use    : make run_base
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "pgm.h"
// tiempo de cuda
#include <cuda_runtime.h>
#include <vector>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Calcular gloID teniendo en cuenta la geometría del grid
  int blockID = blockIdx.x + blockIdx.y * gridDim.x;
  int gloID = blockID * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  if (gloID >= w * h) return; // En caso de hilos extras en el bloque

  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}


//*****************************************************************
int main(int argc, char **argv)
{
  int i;
  int threshhold = argv[2] ? atoi(argv[2]) : 3115;
  // tiempo
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // create a PGMImage object
  PGMImage* inImg = new PGMImage(argv[1]); 

  inImg->setColor(0,255,0);

  int *cpuht;
  int w = inImg->getXDim();
  int h = inImg->getYDim();  

  float *d_Cos;
  float *d_Sin;

   cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
   cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);
 
  // CPU calculation
  CPU_HoughTran(inImg->getPixels(), w, h, &cpuht);

  // Pre-compute values to be stored
  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
  float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
  float rad = 0;

  for (int i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Copiar valores de seno y coseno a la memoria constante
   cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
   cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
   
  // Setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg->getPixels(); // h_in contiene los pixeles de la imagen

  h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

  cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
  cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

  // Execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  // 1 thread por pixel
  int blockNum = ceil(w * h / 256.0);
  dim3 gridDim(blockNum, 1);
  dim3 blockDim(16, 16); // Puedes ajustar estos valores según tus necesidades

  // Lanzar el kernel y medir el tiempo de ejecución
  cudaEventRecord(start, 0);

  GPU_HoughTran<<<gridDim, blockDim>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Tiempo de ejecución del kernel: %f ms\n", elapsedTime);

  // Get results from device
  cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // Compare CPU and GPU results
  for (int i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  std::vector<std::pair<int, int>> lines;                                         // Vector de pares de enteros (r, th)
  for (i = 0; i < degreeBins * rBins; i++){                                       // Itera sobre los bins de angulo y radio
    if (h_hough[i] > threshhold) {                                                 // Si el acumulador de la transformada de Hough es mayor que el umbral
      // pair order: r, th
      int my_r = i / degreeBins;                                                  // Calcula el radio
      int my_th = i % degreeBins;                                                 // Calcula el angulo
      std::pair<int, int> line = {my_r, my_th};                                   // Crea el par de enteros (r, th)
      lines.push_back(line);                                                      // Agrega el par de enteros al vector
    }
  }
  inImg->saveAsJPEG("Base.jpeg", lines, radInc, rBins);
  printf("Done!\n");

  // Liberar memoria
  cudaFree(d_in);
  cudaFree(d_hough);
  cudaFree(d_Cos);
  cudaFree(d_Sin);

  free(h_hough);
  free(pcCos);
  free(pcSin);
  delete[] cpuht;

  return 0;
}