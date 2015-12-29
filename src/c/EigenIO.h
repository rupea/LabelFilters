#ifndef _EIGENIO_H
#define _EIGENIO_H

#include "typedefs.h"   // ActiveDataSet
#include "Eigen/Dense"

#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <ostream>
#include <fstream>
#include <string>

// load_projections loads some file written out by octave, so moved to EigenOctave.h
// (there should be two C++ routines to write and read the projection file)
//void load_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const string& filename, bool verbose = false);

/** save weights, lower- and upper-bounds from \c solve_optimization. */
void write_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const std::string& filename, bool verbose = false);

/** load weights, lower- and upper-bounds from \c solve_optimization. */
void read_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const std::string& filename, bool verbose = false);

void save_LPSR_model(const char* filename, const DenseColM& centers, const ActiveDataSet& active_classes);

int load_LPSR_model(const char* filename, DenseColM& centers, ActiveDataSet*& active_classes);

/** Read \c cols columns each with \c rows rows, starting from column \c start_col.
 *  The file should have a matrix sotred in column major format, with 32 bit float entries. 
 */
void read_binary(const char* filename, DenseColMf& m, const DenseColMf::Index rows, const DenseColMf::Index cols, const DenseColMf::Index start_col = 0);

template <typename Block, typename Alloc>
  void save_bitvector(std::ofstream& out, const boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits = bs.size();
  size_t num_blocks = bs.num_blocks();
  std::vector<Block> blocks(num_blocks);
  to_block_range(bs, blocks.begin());  
  out.write((char*)&num_bits, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)&num_blocks, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)(&(blocks[0])), num_blocks*sizeof(Block));  
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
}

template <typename Block, typename Alloc>
  int load_bitvector(std::ifstream& in, boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits,num_blocks;
  in.read((char*)&num_bits, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  in.read((char*)&num_blocks, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  std::vector<Block> blocks(num_blocks);
  in.read((char*)(&(blocks[0])), num_blocks*sizeof(Block));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  bs.resize(num_bits);
  from_block_range(blocks.begin(), blocks.end(), bs);
  bs.resize(num_bits);
  return 0;
}


#endif
