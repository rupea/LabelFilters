//#include <octave/oct.h> 
//#include <octave/parse.h> 
//#include <octave/oct-map.h>
//#include <octave/builtin-defun-decls.h>
//#include <octave/octave.h>
//#include <octave/parse.h>
//#include <octave/toplev.h>
#include "EigenIO.h"

//#include "printing.hh"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace std;

void read_dense_binary(const char* filename, ovaDenseColM& m, const DenseColMf::Index rows, const DenseColMf::Index cols, const DenseColMf::Index start_col)
{
  assert(sizeof(DenseColMf::Scalar) == 4);
  ifstream in(filename, ios::in | ios::binary);
  if (in.is_open())
    {
      try
	{
	  m.resize(rows,cols);
	}
      catch(std::exception& e)
	{
	  std::cerr << "Error allocating OVA matrix of size " << rows << "x" << cols << ". " << std::endl;
	  throw;
	}
      in.seekg(start_col*rows*sizeof(DenseColMf::Scalar));
      in.read((char*)m.data(), rows*cols*sizeof(DenseColMf::Scalar));
      if(!in)
	{
	  cerr << "Error reading file " << filename << ". Only " << in.gcount() << " bytes read out of " << rows*cols*sizeof(DenseColMf::Scalar) << endl;
	  exit(-1);
	}
      in.close();
    }
  else
    {
      cerr << "Trouble opening file " << filename << endl;
      exit(-1);
    }
}

// reads a sparse matrix stored in Compressed Column (row) format.
// index type is assumed to be 32 bit integers 
// value type is assumed to be 32 bit floats(this could be stored in a header)
// first two index types store the outer and inner dimensions
// Next is an array of outer+1  index types representing the outer index
// Next is an array of index types representing the inner indices
// Next is an array of index types representing the values.

// to do , store the size of the index typte and the size of the scalar type in the file
void read_sparse_binary(const char* filename, ovaSparseColM& m, ovaSparseColM::StorageIndex rows, ovaSparseColM::StorageIndex cols, const ovaSparseColM::StorageIndex start_col /*=0*/)
{
  assert(sizeof(ovaSparseColM::Scalar) == 4);
  assert(sizeof(ovaSparseColM::StorageIndex) == 8);
  ifstream in(filename, ios::in | ios::binary);
  if (in.is_open())
    {
      ovaSparseColM::StorageIndex nouter=0;  // nr of outer dimensions cols in a column major matrix
      ovaSparseColM::StorageIndex ninner=0;   // n of row in a rowmajor matrix      
      in.read((char*)(&nouter), sizeof(ovaSparseColM::StorageIndex));
      in.read((char*)(&ninner), sizeof(ovaSparseColM::StorageIndex));
      if (cols == 0) cols = nouter;
      if (rows == 0) rows = ninner;

      try
	{
	  m.resize(rows,cols);
	}
      catch(std::exception& e)
	{
	  std::cerr << "Error allocating OVA matrix of size " << rows << "x" << cols << ". " << std::endl;
	  throw;
	}

      // read col  outer indices starting from the start_col
      in.seekg((2+start_col)*sizeof(ovaSparseColM::StorageIndex));
      ovaSparseColM::StorageIndex* outerIndexPtr = m.outerIndexPtr();
      in.read((char*)outerIndexPtr, (cols+1)*sizeof(ovaSparseColM::StorageIndex));
      if (!in) 
	{
	  cerr << "Error reading the outer index array from " << filename << ". Only " << in.gcount() << " bytes read out of " << cols*sizeof(ovaSparseColM::StorageIndex);
	  exit(-1);
	}
      //skip the rest of the outer indices
      in.seekg((2+nouter)*sizeof(ovaSparseColM::StorageIndex));
      ovaSparseColM::StorageIndex totalNZ = 0; 
      in.read((char*)(&totalNZ), sizeof(ovaSparseColM::StorageIndex));
      //read the inner indices 
      ovaSparseColM::StorageIndex nnz = outerIndexPtr[cols]-outerIndexPtr[0];
      m.resizeNonZeros(nnz);
      in.seekg((2+nouter+1+outerIndexPtr[0])*sizeof(ovaSparseColM::StorageIndex));
      in.read((char*)m.innerIndexPtr(), nnz*sizeof(ovaSparseColM::StorageIndex));
      if (!in) 
	{
	  cerr << "Error reading the inner index array from " << filename << ". Only " << in.gcount() << " bytes read out of " << nnz*sizeof(ovaSparseColM::StorageIndex);
	  exit(-1);
	}
      // skip over the last innerindices and the first values until getting to the desired chunk
      in.seekg((2+nouter+1+totalNZ)*sizeof(ovaSparseColM::StorageIndex)+outerIndexPtr[0]*sizeof(ovaSparseColM::Scalar));
      //read the values
      in.read((char*)m.valuePtr(), nnz*sizeof(ovaSparseColM::Scalar));
      if (!in) 
	{
	  cerr << "Error reading the values array from " << filename << ". Only " << in.gcount() << " bytes read out of " << nnz*sizeof(ovaSparseColM::Scalar);
	  exit(-1);
	}
      // set the outerindices to start form 0. Do it in reverse order so that outerIndexPtr[0] is not overwritten
      // until the end
      for (ovaSparseColM::StorageIndex i=cols; i>=0; i--)
	{	  
	  outerIndexPtr[i] -= outerIndexPtr[0];
	}
      in.close();
    }
  else
    {
      cerr << "Trouble opening file " << filename << endl;
      exit(-1);
    }
}

/*
void save_LPSR_model(const char* filename, const DenseColM& centers, const ActiveDataSet& active_classes)
{
  ofstream out(filename, ios::out|ios::binary);
  int k = centers.cols();
  size_t dim = centers.rows();
  if (out.is_open())
    {
      out.write((char*)&k, sizeof(int));
      out.write((char*)&dim, sizeof(size_t));
      if (out.fail())
	{
	  cerr << "Error writing file " << filename << endl;
	}
      out.write((char*)centers.data(), centers.size()*sizeof(DenseColM::Scalar));
      if (out.fail())
	{
	  cerr << "Error writing file " << filename << endl;
	}
    }
  else
    {
	  cerr << "Error writing file " << filename << endl;
    }      
  cout << "saved centers" << endl;
  cout << active_classes.size() << endl;
  for (int i=0;i<k;i++)
    {
      save_bitvector(out,*(active_classes[i]));
    }    
  out.close();
}


int load_LPSR_model(const char* filename, DenseColM& centers, ActiveDataSet*& active_classes)
{
  int k;
  size_t dim;
  ifstream in(filename, ios::in|ios::binary);
  if (in.is_open())
    {
      in.read((char*)&k, sizeof(int));
      if (in.fail())
	{
	  cerr << "Error reading file " << filename << endl;
	  return -1;
	}
      in.read((char*)&dim, sizeof(size_t));
      if (in.fail())
	{
	  cerr << "Error reading file " << filename << endl;
	  return -1;
	}
      centers.resize(dim,k); 
      in.read((char*)centers.data(), dim*k*sizeof(DenseColM::Scalar));
      if (in.fail())
	{
	  cerr << "Error reading file " << filename << endl;
	  return -1;
	}     
    }
  else
    {
      cerr << "Error reading file " << filename << endl;
      return -1;
    }        
  active_classes = new ActiveDataSet(k);
  for (int i=0;i<k;i++)
    {
      active_classes->at(i) = new boost::dynamic_bitset<>();
      if (load_bitvector(in,*(active_classes->at(i))) != 0)
	{
	  return -1;
	}
    }  
  in.close();
  return 0;
}
*/
