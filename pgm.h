#ifndef PGM_H
#define PGM_H
#include <vector>
class PGMImage
{
private:
  int x_dim;
  int y_dim;
  int num_colors;
  unsigned char *pixels;
  struct RGB
  {
    int r, g, b;
  } color;

public:
  PGMImage(char *fname);
  PGMImage(int x, int y, int col);
  ~PGMImage(void);
  void setColor(int r, int g, int b);
  int getXDim(void);
  int getYDim(void);
  unsigned char *getPixels(void);
  void saveImg(const char *destFile, std::vector<std::pair<int, int>> highlightLines, float angleStep, int radiusDivisions);
};

#endif
