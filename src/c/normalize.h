#ifndef __NORMALIZE_H
#define __NORMALIZE_H

#include "typedefs.h"

// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_col(SparseM& mat);

void normalize_col(DenseM& mat);

void normalize_row(SparseM &mat);

#endif


