#ifndef __FILTER_H
#define __FILTER_H

#include "Eigen/Dense"
#include <boost/dynamic_bitset.hpp>
#include <vector>

/** Filter with fast random-access by individual projection values.
 * - Best used when 
 *   - need to evaluate projections serially in random order, or
 *   - number of calls to \c Filter::filter(double) is >> number of classes.
 */
class Filter
{
public:
    /** For a specific projection dimension, create lookup table of bitmaps for possible classes.
     * - \c l and \c u are lower and upper bounds for each class. */
    Filter(const Eigen::VectorXd& l, const Eigen::VectorXd& u);
    ~Filter();

    /** Given a projection value, \c xproj, what classes are possible?
     * \return bitset pointer \b unusable after \c Filter destructor runs.
     */
    boost::dynamic_bitset<> const* filter (double xproj) const;
private:  
    void init_map(std::vector<int>& ranks); ///< construction helper

    Eigen::VectorXd _sortedLU; ///< sort the concatenation of all (lower,upper) values
    /** Tabulate {classes} given where projection value falls in \c _sortedLU.
     * - As projection values pass l/u boundaries in \c _sortedLU,
     * - one bit in the class bitmap changes.
     * - So save these easily-constructed bitmaps. */
    std::vector<boost::dynamic_bitset<> > _map;
};

inline const boost::dynamic_bitset<>* Filter::filter(double xproj) const
{
    return &_map[std::distance( _sortedLU.data(),
                                std::lower_bound(_sortedLU.data(),
                                                 _sortedLU.data()+_sortedLU.size(),
                                                 xproj))];
}

#if 1-1
/** Filter with fast access for a batch of projection values.
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
    std::vector<boost::dynamic_bitset<>>* filter (vector<double> xproj) const;
private:
    boost::dynamic_bitset<> bs;
};
#endif


#endif
