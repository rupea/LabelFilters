#include <vector>
#include "boolmatrix.h"

using namespace std;

boolmatrix::boolmatrix(size_t n, size_t m)
{
  _data = vector<bool>(n*m,false);
  _nrow = n;
  _ncol = m;
}
