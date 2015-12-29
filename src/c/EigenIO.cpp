//#include <octave/oct.h> 
//#include <octave/parse.h> 
//#include <octave/oct-map.h>
//#include <octave/builtin-defun-decls.h>
//#include <octave/octave.h>
//#include <octave/parse.h>
//#include <octave/toplev.h>
#include "EigenIO.h"
#include "typedefs.h"
//#include "EigenOctave.h"
#include "Eigen/Sparse"
#include <iostream>
#include <ostream>
#include <fstream>
#include <stdlib.h>
#include <string>

using namespace std;

void read_binary(const char* filename, DenseColMf& m, const DenseColMf::Index rows, const DenseColMf::Index cols, const DenseColMf::Index start_col)
{
  assert(sizeof(DenseColMf::Scalar) == 32);
  ifstream in(filename, ios::in | ios::binary);
  if (in.is_open())
    {
      m.resize(rows,cols);
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
