#ifndef __PREDICT_H
#define __PREDICT_H

#include "typedefs.h"
#include "PredictionSet.h"
#include "filter.h"

/** Given col-wise projection vectors, return the final bitmap of allowed classes.
 * \p no_active sum of <B>1</B>s count of final projection bitmap over all examples.<\br>
 *              Indicative of amount of remaining classification work to do.
 * \p projections row-wise matrix of projections, n x p
 * \p lmat      lower bounds, per class
 * \p umat      upper bounds, per class
 * \p verbose   show messages about Filter construction
 *
 * - constructs p Filters (fast bitmap lookup), one for each projection axis
 * - applying a Filter to the projection of an example gives a bitset
 *   - with <B>1</B>s for each allowed class
 * - \c getactive applies all Filters, sequentially <B>AND</B>ing the
 *   projection Filter bitmaps into a returned class bitmap
 *
 * \return vector of possible-class bitmaps, one for each row-wise example in \c x.
 * \detail
 * This is actually a more fundamental operation than \c getactive !!
 */
ActiveDataSet* projectionsToActiveSet( VectorXsz& no_active, DenseM const& projections,
                                       const DenseColM& lmat, const DenseColM& umat,
                                       bool verbose);



/** Given col-wise projection vectors, update the bitmap of allowed classes.
 * \p active  bitmap of allowed classes to be updated. If it is nullptr it will be initialized.
 * \p f       a Filter object (fast bitmap lookup) used to update the active classes. 
 *
 * - applying a Filter to the projection of an example gives a bitset
 *   - with <B>1</B>s for each allowed class
 *
 * \return the total number of active classes in active (the total number of bits set to 1)
 */
size_t update_active(ActiveDataSet** active, Filter const& f, VectorXd const&  proj);

/** For each row-wise example in matrix \c x, calculate the class filter.
 * \p no_active sum of <B>1</B>s count of final projection bitmap over all examples.<\br>
 *              Indicative of amount of remaining classification work to do.
 * \p x         example matrix ~ row-wise d-dim training examples, n x d
 * \p wmat      projection matrix ~ col-wise projection vectors, d x p
 * \p lmat      lower bounds, per class
 * \p umat      upper bounds, per class
 * \p verbose   show messages about Filter construction
 *
 * - projects every example onto projections
 * - constructs p Filters (fast bitmap lookup), one for each projection axis
 * - applying a Filter to the projection of an example gives a bitset
 *   - with <B>1</B>s for each allowed class
 * - \c getactive applies all Filters, sequentially <B>AND</B>ing the
 *   projection Filter bitmaps into a returned class bitmap
 *
 * \return vector of possible-class bitmaps, one for each row-wise example in \c x.
 *
 * transform x={examples} + MCsoln{w,l,u} into an ActiveDataSet.
 * - ActiveDataSet is a vector-of-ptr-to-bitset.
 * - one bitset is produced per projection dimension
 */
template <typename Eigentype>
ActiveDataSet* getactive( VectorXsz& no_active, const Eigentype& x,
                          const DenseColM& wmat, const DenseColM& lmat, const DenseColM& umat,
                          bool verbose = false);

/** predict, w/o PredictionSet.
 * \p w is a set of linear projection lines
 */
template <typename Eigentype>
PredictionSet* predict ( Eigentype const& x, DenseColMf const& w,
                         ActiveDataSet const* active, size_t& nact,
                         bool verbose             = false,
                         predtype keep_thresh     = boost::numeric::bounds<predtype>::lowest(),
                         size_t keep_size         = boost::numeric::bounds<size_t>::highest(),
                         size_t const start_class = 0);

/** predict, with PredictionSet.
 * \p w is a set of linear projection lines
 */
template <typename Eigentype>
void predict( PredictionSet* predictions,
              Eigentype const& x, DenseColMf const& w,
              ActiveDataSet const* active, size_t& nact,
              bool verbose             = false,
              predtype keep_thresh     = boost::numeric::bounds<predtype>::lowest(),
              size_t keep_size         = boost::numeric::bounds<size_t>::highest(),
              size_t const start_class = 0);


#endif
