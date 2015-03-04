#ifndef _EIGENIO_H
#define _EIGENIO_H

void load_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const string& filename, bool verbose = false);

void read_binary(const char* filename, DenseColMf& m, const DenseColMf::Index rows, const DenseColMf::Index cols, const DenseColMf::Index start_col = 0);


#endif
