/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
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
