#ifndef MCFILTER_H
#define MCFILTER_H
/** \file
 * implementation of the mc filter. 
 */
#include "mcsoln.h"
#include "filter.h" 
#include <fstream>

class MCfilter : protected MCsoln
{
 public:
  
  //    MCfilter( char const* const solnfile = nullptr );
  MCfilter();
  MCfilter(MCsoln const& s);
  ~MCfilter(){}
  
  //    param_struct const& getParms() const {return this->parms;}
  MCsoln       const& getSoln()  const {return *this;}
  //    MCsoln            & getSoln()        {return *this;}
  void read( std::string const& fname )
  {						
    std::ifstream is(fname);
    read(is);
    is.close();
  }
  
  void read( std::istream& is )
  {       
    MCsoln::read(is);
    init_filters();
  }
  void write( std::ostream& os, enum Fmt fmt=BINARY) const
  {
    MCsoln::write(os,fmt);
  }
  
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
    void filter(/*out*/ std::vector<boost::dynamic_bitset<>>& active, /*in*/ EIGENTYPE const& x, int np = 0);
 private:
  std::vector<Filter> _filters;
  void init_filters();
  //    int getNthreads( param_struct const& params ) const;
};


#endif //MCFILTER_H
