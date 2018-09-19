#ifndef __FILTER_H
#define __FILTER_H

#include "Eigen/Dense"
#include "roaring.hh"
#include <vector>

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

    /** debug... */
    ssize_t idx(double xproj) const{
        ssize_t ret = std::distance( _sortedLU.data(),
                                    std::lower_bound(_sortedLU.data(),
                                                     _sortedLU.data()+_sortedLU.size(),
                                                     xproj
                                                     ));
        return ret;
    }
private:
    void init_map(std::vector<int>& ranks); ///< construction helper

    Eigen::VectorXd _sortedLU; ///< sort the concatenation of all (lower,upper) values
    /** Tabulate {classes} given where projection value falls in \c _sortedLU.
     * - As projection values pass l/u boundaries in \c _sortedLU,
     * - one bit in the class bitmap changes.
     * - So save these easily-constructed bitmaps. */
    std::vector<Roaring> _map;
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

#if 0
/** TO IMPLEMENT: Filter with fast access for a batch of projection values.
 * - Best used when
 *   - we have a "batch" of values to process, just once, and
 *   - number of calls to \c Filter::filter(double) is << number of classes.
 * \detail
 *   - sort the projection values
 *   - and bitmap can be modified as we ascend the projection values.
 *     - and we can stop whenever we have no more projection values.
 */
class FilterBatch
{
    FilterBatch(const Eigen::VectorXd& l, const Eigen::VectorXd& u);
    ~FilterBatch();
    /** Given a projection value, \c xproj, what classes are possible?
     * \return vector of bitsets, <b>up to client to \c delete returned pointer</b> */
    std::vector<Roaring>* filter (vector<double> xproj) const;
private:
    Eigen::VectorXd _sortedLU;
};
#endif


#endif
