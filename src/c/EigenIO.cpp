#include <octave/oct.h> 
#include <octave/parse.h> 
#include <octave/oct-map.h>
#include <octave/builtin-defun-decls.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include "EigenOctave.h"
#include "EigenIO.h"




// TO DO: put protections when files are not available or the right 
// variables are not in them.
// now it crashes badly with a seg fault and can corrupt other processes
void load_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const string& filename, bool verbose)
{
  octave_value_list args; 
  args(0) = filename; // the projection filename 
  args(1) = "w";
  args(2) = "min_proj";
  args(3) = "max_proj";
  if (verbose)
    {
      cout << "Loading file " << args(0).string_value() << " ... " <<endl;
    }
  octave_value_list loaded = Fload(args, 1);
  //feval("load", args, 0); // no arguments returned 
  if (verbose)
    {
      cout << "success" << endl; 
    }
  
  wmat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(1).string_value()).array_value());
  lmat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(2).string_value()).array_value());
  umat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(3).string_value()).array_value());
  args.clear();
  loaded.clear();
  Fclear();
}

// the file should have a matrix sotred in column major format, with 32 bit float entries. 
// read cols columns each with rows rows. Start from column start_col 
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
