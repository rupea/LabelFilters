#ifndef __FIND_W_H
#define __FIND_W_H

#include "typedefs.h"
#include "parameter.h"
#include "mutexlock.h"  // deprecate this XXX (use std::mutex?)
//#include "mcsolver.h"   // maybe just the fwd declarations?
#include <iosfwd>
#include  <array>

/**
 * Solve the optimization using the gradient descent on hinge loss.
 *
 * - Suppose \c d-dimensional features, \c k classes, and \c p features
 * - The training data \c x is a <examples> x \c d matrix [k cols]
 * - The training labels \c y is a <examples> x \c k matrix [k cols],
 *   - typically a vector converted to a SparseMb (sparse bool matrix)
 *  \p weights          inout: d x p matrix, init setRandom
 *  \p lower_bounds     inout: k x p matrix, init setZero
 *  \p upper_bounds     inout: k x p matrix, init setZero
 *  \p objective_val    out:  VectorXd(k...) (a growing history)
 *  \p w_avg            inout: d x p matrix, init setRandom
 *  \p l_avg            inout: k x p matrix, init setZero
 *  \p u_avg            inout: k x p matrix, init setZero
 *  \p object_val_avg   out: VectorXd(k) (a growing history)
 *  \p x                in: X x d, X d-dim training examples
 *  \p y                in: X x 1, X labels of each training example
 *  \p params           many parameters, eg from set_default_params()
 *
 *  - The library has instantiations for EigenType:
 *    - DenseM and
 *    - SparseM
 */

template<typename EIGENTYPE>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, Eigen::VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, Eigen::VectorXd& objective_val_avg,
                        EIGENTYPE const& x,
                        SparseMb const& y,
                        param_struct const& params);

#if 1 // proposed for lua api.



/** lazy stats for x,y data. WIP, so opaque for now. */
struct MCLazyData;


#endif // proposed
#endif // __FIND_W_H
