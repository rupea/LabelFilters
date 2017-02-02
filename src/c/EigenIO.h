#ifndef _EIGENIO_H
#define _EIGENIO_H

#include "typedefs.h"   // ActiveDataSet and Eigen dense and sparse support

#include <iostream>

void save_LPSR_model(const char* filename, const DenseColM& centers, const ActiveDataSet& active_classes);
int load_LPSR_model(const char* filename, DenseColM& centers, ActiveDataSet*& active_classes);

/** Read \c cols columns each with \c rows rows, starting from column \c start_col.
 *  The file should have a matrix sotred in column major format, with 32 bit float entries. 
 */
void read_dense_binary(const char* filename, ovaDenseColM& m, const DenseColMf::Index rows, const DenseColMf::Index cols, const DenseColMf::Index start_col = 0);


void read_sparse_binary(const char* filename, ovaSparseColM& m, const ovaSparseColM::StorageIndex rows, const ovaSparseColM::StorageIndex cols, const ovaSparseColM::StorageIndex start_col = 0);
#endif
