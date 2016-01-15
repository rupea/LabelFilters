#ifndef __PREDICT_H
#define __PREDICT_H

#include "typedefs.h"
#include "PredictionSet.h"

/** transform x={examples} + MCsoln{w,l,u} into an ActiveDataSet.
 * - ActiveDataSet is a vector-of-ptr-to-bitset.
 * - one bitset is produced per projection dimension */
template <typename Eigentype>
ActiveDataSet* getactive( VectorXsz& no_active, const Eigentype& x,
                          const DenseColM& wmat, const DenseColM& lmat, const DenseColM& umat,
                          bool verbose = false);

/** predict, w/o PredictionSet */
template <typename Eigentype>
PredictionSet* predict ( Eigentype const& x, DenseColMf const& w,
                         ActiveDataSet const* active, size_t& nact,
                         bool verbose             = false,
                         predtype keep_thresh     = boost::numeric::bounds<predtype>::lowest(),
                         size_t keep_size         = boost::numeric::bounds<size_t>::highest(),
                         size_t const start_class = 0);

/** predict, with PredictionSet */
template <typename Eigentype>
void predict( PredictionSet* predictions,
              Eigentype const& x, DenseColMf const& w, 
              ActiveDataSet const* active, size_t& nact,
              bool verbose             = false,
              predtype keep_thresh     = boost::numeric::bounds<predtype>::lowest(),
              size_t keep_size         = boost::numeric::bounds<size_t>::highest(),
              size_t const start_class = 0);


#endif
