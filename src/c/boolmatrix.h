#ifndef __BOOLMATRIX_H
#define __BOOLMATRIX_H

using namespace std;

class boolmatrix
{
 public:
  boolmatrix(size_t n, size_t m);
  inline bool get_val(size_t i, size_t j) const
  {
    return (_data[i*_ncol + j]);
  }
  inline void set_val(size_t i, size_t j, bool val)
  {
    _data[i*_ncol + j] = val;
  }
  
 private:
  vector<bool> _data;
  size_t _nrow;
  size_t _ncol;

};


#endif // __BOOLMATRIX_H
