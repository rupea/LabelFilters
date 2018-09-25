/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCFILTER_H
#define MCFILTER_H
/** \file
 * implementation of the mc filter. 
 */
#include "mcsoln.h"
#include "filter.h"
#include "typedefs.h"
#include <fstream>

class MCfilter : protected MCsoln
{
 public:
  
  MCfilter();
  MCfilter(MCsoln const& s);
  ~MCfilter();
  
  MCsoln       const& getSoln()  const {return *this;}
  void read( std::string const& fname )
  {						
    std::ifstream is(fname);
    if(!is.good()) 
      { 
	std::string errmsg = "trouble opening solution file ";
	errmsg += fname;	
	throw std::runtime_error(errmsg);
      }
    read(is);
    is.close();
  }
  
  inline void read( std::istream& is ){ MCsoln::read(is);init_filters();}
  inline void write( std::ostream& os, enum Fmt fmt=BINARY) const {MCsoln::write(os,fmt);}  
  inline bool isempty()const {return !_filters.size();}
  inline uint32_t nFilters() const {return _filters.size();}
  
  /** apply the MC filter on data 
   * \p active  bit matrix indicating if a class is active or has been filtered out 
   *              active(i,k) = 1 if class k is active for example i
   * \p x       data, row-wise examples of dimension MCsoln::d
   * \p np   number of filters to apply (0 = apply all filters)
   * \internal
   *
   * - NOTE: EIGENTYPE \c DenseM and \c SparseM are provided by the default library.
   *   - Please only include \c find_w_detail.hh for \em strange 'x' types.
   */
  template< typename EIGENTYPE >
    void filter(/*out*/ ActiveSet& active, /*in*/ EIGENTYPE const& x, int np = 0) const;
 private:
  std::vector<Filter*> _filters;
  void init_filters();
  void delete_filters();
};


#endif //MCFILTER_H
