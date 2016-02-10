#ifndef MCPREDICT_HH
#define MCPREDICT_HH
#include "mcpredict.h"
#include "typedefs.h"
//#include <boost/dynamic_bitset.hpp>

/** non-templated workhorse.
 * \p active            output [ nExamples x nClass ] bitsets
 * \p projections       input [ nProj x nProj ] row-wise example projection values.
 *                      NB: \b Transpose of x * weights_avg
 * \p s                 class {l,u} bounds used to evaluate projection values.
 *
 * \sa getactive
 * \sa predict.h
 *
 * \note \ref predict.cpp maintains an expensive no_active[i].  I don't
 *       understand what use this sum of intermediate count() values has.
 *       You can always use \c active[i].count() (slow) to
 *       get the \em final number of remaining classes.
 */
void projectionsToBitsets( DenseM const& projections,
                           MCsoln const& s,
                           std::vector<boost::dynamic_bitset<>>& active );

template<typename EigenType> inline
std::vector<boost::dynamic_bitset<>> project( EigenType const& x, MCsoln const& s )
{
    assert( x.cols() == s.weights_avg().rows() );
    DenseM const projections = (x * s.weights_avg).transpose();
    std::vector<boost::dynamic_bitset<>> active;
    projectionsToBitsets( projections, s, active );
}

#endif // MCPREDICT_HH
