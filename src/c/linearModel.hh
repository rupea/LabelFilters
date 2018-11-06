/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __LINEARMODEL_HH
#define __LINEARMODEL_HH

#include "linearModel_detail.hh"


template <typename Eigentype> inline
size_t linearModel::predict (PredictionSet& predictions, Eigentype const& x, ActiveSet const* feasible, bool verbose, predtype keep_thresh, size_t keep_size) const
{
  if (denseOk) 
    {
      return linearmodel_detail::predict(predictions, x, WDense, intercept, feasible, verbose, keep_thresh, keep_size);
    }
  else if (sparseOk)
    {
      return linearmodel_detail::predict(predictions, x, WSparse, intercept, feasible, verbose, keep_thresh, keep_size);
    }
  else
    {
      throw std::runtime_error("LinearModel: predict called without a valid model");
    }
}
#endif //__LINEAEMODEL_HH
