/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __FILTER_H
#define __FILTER_H

#include "Eigen/Dense"
#include "roaring.hh"
#include "typedefs.h"
#include <vector>
#include <list>
/** Filter with fast random-access by individual projection values.
 * - Best used when
 *   - need to evaluate projections serially in random order, or
 *   - number of calls to \c Filter::filter(double) is >> number of classes.
 * - \b Margin
 *   - treatment of projection values <em>exactly equal</em> to a boundary
 *     is not quite right:
 *     - I believe with operator< for lower_bound it returns the bitmap
 *       for projection value \em p if it is range (lower,upper].
 *   - This means that if you \c Filter using a \b zero-margin projection
 *     of training data (i.e. exact lower and upper bounds over all training
 *     data), data points that fall \em precisely on an {l,u}
 *     class boundary may \b incorrectly be rejected.
 */

struct MutexType;

class Filter
{
public:
    /** 0=quiet    1=warnings [suggested]    2=debug */
    static int const verbose;

    /** For a specific projection dimension, create lookup table of bitmaps for possible classes.
     * - \c l and \c u are lower and upper bounds for each class [nClass]. */
    Filter(const Eigen::VectorXd& l, const Eigen::VectorXd& u);
    ~Filter();

    /** Given a projection value, \c xproj, what classes are possible?
     * - the bitset has \b 1 (true) for each possible class.
     * \return bitset pointer \b unusable after \c Filter destructor runs.
     */
    Roaring const* filter (double xproj) const;
    void filter(double xproj, std::list<int>& active) const;
    void filterBatch (Eigen::VectorXd proj, ActiveSet& active, std::vector<MutexType>& mutex) const;

    /** debug... */
    ssize_t idx(double xproj) const{
        ssize_t ret = std::distance( _sortedLU.data(),
                                    std::lower_bound(_sortedLU.data(),
                                                     _sortedLU.data()+_sortedLU.size(),
                                                     xproj
                                                     ));
        return ret;
    }
    /** Tabulate {classes} given where projection value falls in \c _sortedLU.
     * - As projection values pass l/u boundaries in \c _sortedLU,
     * - one bit in the class bitmap changes.
     * - So save these easily-constructed bitmaps. */
    void init_map(); 
private:
    Eigen::VectorXd _l;
    Eigen::VectorXd _u;
    Eigen::VectorXd _sortedLU; ///< sort the concatenation of all (lower,upper) values
    std::vector<int> _sortedClasses; ///< used to recover the classes that coresponds to each in _sortedLU. 
    /** Tabulate {classes} given where projection value falls in \c _sortedLU.
     * - As projection values pass l/u boundaries in \c _sortedLU,
     * - one bit in the class bitmap changes.
     * - So save these easily-constructed bitmaps. */
    std::vector<Roaring> _map;
    bool _mapok;
#ifndef NDEBUG
    mutable uint64_t nCalls;
#endif
};

/** \detail
 * - Note that lower_bound returns 1st item 'not less than'.
 * - so large, -ve \c xproj will return _map[0]
 * - This means _map[0] (lower than lowest lmat left boundary)
 *   must correspond to the empty bitmap (no possible classes)
 * - Considering highest right boundary, a larger xproj will
 *   yield _sortedLU.end(), so _map.size() must be 2*noClasses+1.
 */
inline const Roaring* Filter::filter(double xproj) const
{
#ifndef NDEBUG
    ++nCalls;
#endif
    return &_map[std::distance( _sortedLU.data(),
                                std::lower_bound(_sortedLU.data(),
                                                 _sortedLU.data()+_sortedLU.size(),
                                                 xproj))];
}


// very slow. Do optimized. 
inline void Filter::filter(double xproj, std::list<int>& active) const
{
#ifndef NDEBUG
  ++nCalls;
#endif
  std::list<int>::iterator it = active.begin();
  while (it != active.end())
    {
      (xproj < _l[*it] || xproj > _u[*it])?(it=active.erase(it)):++it;
    }    
}

#endif
