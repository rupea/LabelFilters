#ifndef __LINEARMODEL_DETAIL_H
#define __LINEARMODEL_DETAIL_H


#include "typedefs.h"
#include <boost/numeric/conversion/bounds.hpp>
//#include <boost/limits.hpp>

class PredictionSet;

namespace linearmodel_detail
{
  template <typename Eigentype, typename ovaType>
    std::size_t predict( PredictionSet& predictions,  // output
			 Eigentype const& x, ovaType const& w, Eigen::RowVectorXd const& intercept,
			 ActiveSet const* feasible,
			 bool verbose             = false,
			 predtype keep_thresh     = boost::numeric::bounds<predtype>::lowest(),
			 size_t keep_size         = boost::numeric::bounds<size_t>::highest());
}

#endif  //__LINEARMODEL_DETAIL_H
