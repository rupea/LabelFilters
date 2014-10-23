#ifndef __BOOLMATRIX_H
#define __BOOLMATRIX_H

#include <boost/dynamic_bitset.hpp>

using namespace boost;

class boolmatrix
{
 public:
  boolmatrix(size_t n, size_t m);
  bool get(size_t i, size_t j) const;
  //set the i,j bit to true; more eficient
  void set(size_t i, size_t j);
  // set teh i,j bit to val
  void set(size_t i, size_t j, bool val);
  size_t count() const {return _count;}

 private:
  dynamic_bitset<>* _data;
  size_t _nrow;
  size_t _ncol;
  size_t _count;
};

inline boolmatrix::boolmatrix(size_t n, size_t m)
{
  _data = new dynamic_bitset<> (n*m);
  _nrow = n;
  _ncol = m;
  _count = 0;
}

inline bool boolmatrix::get(size_t i, size_t j) const
{
  return (_data->test(i*_ncol + j));
}
//set the i,j bit to val and return the old value
inline void boolmatrix::set(size_t i, size_t j)
{
  _count += !_data->test_set(i*_ncol + j);
}

inline void boolmatrix::set(size_t i, size_t j, bool val)
{
  bool prev = _data->test_set(i*_ncol + j, val);
  _count += val?(!prev):-(prev);
}

#endif // __BOOLMATRIX_H
