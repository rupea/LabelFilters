/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef PARAMETER_ARGS_H
#define PARAMETER_ARGS_H

#include "parameter.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct MCsolveArgs {
  MCsolveArgs();                          ///< construct empty w/ default parms
  MCsolveArgs(po::variables_map const& vm);

  static po::options_description getDesc();
  
  void extract(po::variables_map const& vm);

  
  SolverParams params;
  std::string prev_soln_file;
};

struct MCxyDataArgs {
  static po::options_description getDesc();
  
  MCxyDataArgs();
  MCxyDataArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> xFiles;      ///< x data file name (io via ???)
  std::vector<std::string> yFiles;      ///< optional. If present, same length as xFiles.
  std::string normData;
  uint rmRareF; /*=0*/  // remove features with fewer than rmRareF non-zero entries.   
  uint rmRareL; /*=0*/  // remove lables with fewer than rmRareL non-zero entries.   
  bool xnorm/*=false*/;   ///< normalize x dims across examples to mean and stdev of 1.0
  int center; ///< center the data when col-normalizing. -1: center for dense data don't center for sparse data. 0: don't center, 1:center. (Default -1) 
  bool xunit/*=false*/;   ///< normalize each x example to unit length
  double xscale;          ///< multiply each x example by a constant
  
};

struct MCprojectorArgs {      
  static po::options_description getDesc();
  
  MCprojectorArgs();
  MCprojectorArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> lfFiles;   ///< files to read the trained label filters from
    
  std::vector<int> nProj;    ///< numbers of filters (projections) to use. -1 means use all filters. -2 means {0,1,2,...,no_fiters}
};

struct MCclassifierArgs{
  static po::options_description getDesc();
  
  MCclassifierArgs();
  MCclassifierArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> modelFiles;
    
  double keep_thresh;
  uint32_t keep_top;
  double threshold;
  uint32_t min_labels;
  
};  
#endif // PARAMETER_ARGS_H
