/*/============================================================================
 Author        : Juan Angel Carrera, Juan Carlos Bajan, Jose Mariano Reyes
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <jpeglib.h>
#include <math.h>
#include <vector>
#include "pgm.h"

using namespace std;

// representar imágenes en formato PGM
PGMImage::PGMImage(char *fname)
{
   this->color = {0, 0, 0}; // color inicializado a negro
   this->x_dim = this->y_dim = this->num_colors = 0;
   this->pixels = NULL;

   FILE *ifile;
   ifile = fopen(fname, "rb");
   if (!ifile)
      return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &this->x_dim, &this->y_dim, &this->num_colors);

   getline((char **)&buff, &temp, ifile); // eliminar CR-LF

   assert(this->x_dim > 1 && this->y_dim > 1 && this->num_colors > 1);
   this->pixels = new unsigned char[this->x_dim * this->y_dim];
   fread((void *)this->pixels, 1, this->x_dim * this->y_dim, ifile);

   fclose(ifile);
}

// darle color a las lineas por dibujar
void PGMImage::setColor(int r, int g, int b)
{
   this->color = {r, g, b};
}

// constructor para imagen nueva
PGMImage::PGMImage(int x = 100, int y = 100, int col = 16)
{
   this->num_colors = (col > 1) ? col : 16;
   this->x_dim = (x > 1) ? x : 100;
   this->y_dim = (y > 1) ? y : 100;
   this->pixels = new unsigned char[x_dim * y_dim];
   memset(this->pixels, 0, this->x_dim * this->y_dim);
   assert(this->pixels);
}

// cuando se elimina la imagen
PGMImage::~PGMImage()
{
   if (this->pixels != NULL)
      delete[] this->pixels;
   this->pixels = NULL;
}

// consigue el tamano x de la imagen
int PGMImage::getXDim(void)
{
   return this->x_dim;
}

// consigue el tamano y de la imagen
int PGMImage::getYDim(void)
{
   return this->y_dim;
}

// retorna los pixeles de la imagen
unsigned char *PGMImage::getPixels(void)
{
   return this->pixels;
}

// Función auxiliar para colorear un pixel en la imagen RGB
void setRGBPixel(unsigned char *data, int index, unsigned char r, unsigned char g, unsigned char b)
{
   data[index * 3] = r;
   data[index * 3 + 1] = g;
   data[index * 3 + 2] = b;
}

// guarda la imagen en el nuevo formato
void PGMImage::saveImg(const char *destFile, std::vector<std::pair<int, int>> highlightLines, float angleStep, int radiusDivisions)
{
   // Preparación de variables para el cálculo
   float diagonal = sqrt(pow(this->x_dim, 2) + pow(this->y_dim, 2)) / 2;
   float radiusStep = diagonal * 2 / radiusDivisions;
   int centerX = this->x_dim / 2, centerY = this->y_dim / 2;

   // Crear imagen RGB
   auto *rgbData = new unsigned char[this->x_dim * this->y_dim * 3];

   // Procesar píxeles
   for (int y = 0; y < this->y_dim; ++y)
   {
      for (int x = 0; x < this->x_dim; ++x)
      {
         int index = y * this->x_dim + x;
         bool markedLine = false;

         // Verificar si el pixel está en alguna de las líneas seleccionadas
         for (const auto &[radiusIndex, angleIndex] : highlightLines)
         {
            float radius = radiusIndex * radiusStep - diagonal;
            float angle = angleIndex * angleStep;

            if (fabs(radius - (x - centerX) * cos(angle) - (centerY - y) * sin(angle)) < 0.5)
            {
               markedLine = true;
               break;
            }
         }

         // Colorear el pixel
         if (markedLine)
         {
            setRGBPixel(rgbData, index, this->color.r, this->color.g, this->color.b);
         }
         else
         {
            setRGBPixel(rgbData, index, pixels[index], pixels[index], pixels[index]);
         }
      }
   }

   // Configuración de la compresión JPEG
   struct jpeg_compress_struct cinfo;
   struct jpeg_error_mgr jerr;

   cinfo.err = jpeg_std_error(&jerr);
   jpeg_create_compress(&cinfo);

   FILE *outfile = fopen(destFile, "wb");
   if (!outfile)
   {
      delete[] rgbData;
      return;
   }

   jpeg_stdio_dest(&cinfo, outfile);

   cinfo.image_width = this->x_dim;
   cinfo.image_height = this->y_dim;
   cinfo.input_components = 3;
   cinfo.in_color_space = JCS_RGB;

   jpeg_set_defaults(&cinfo);
   jpeg_set_quality(&cinfo, 75, TRUE);

   jpeg_start_compress(&cinfo, TRUE);

   JSAMPROW row_pointer[1];
   while (cinfo.next_scanline < cinfo.image_height)
   {
      row_pointer[0] = &rgbData[cinfo.next_scanline * this->x_dim * 3];
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
   }

   jpeg_finish_compress(&cinfo);
   fclose(outfile);
   jpeg_destroy_compress(&cinfo);

   delete[] rgbData;
}
