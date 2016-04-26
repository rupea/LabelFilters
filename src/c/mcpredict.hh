#ifndef MCPREDICT_HH
#define MCPREDICT_HH
#include "mcpredict.h"
#include "typedefs.h"
#include "printing.hh" //prettyDims
#include <iostream>
#include <iomanip>
//#include <boost/dynamic_bitset.hpp>

/** non-templated workhorse.
 * \p active            output [ nExamples x nClass ] bitsets
 * \p projections       input [ nProj x nClass ] row-wise example projection values.
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
    using namespace std;
    assert( x.cols() == s.weights_avg.rows() );
    DenseM const projections = (x * s.weights_avg).transpose();
    cout<<"project(x,MCsoln): x"<<prettyDims(x)<<", w"<<prettyDims(s.weights_avg)<<", projections"<<prettyDims(projections)<<endl;
    if(0){
        for(uint32_t i=0; i<x.rows(); ++i){
            {
                ostringstream oss;
                oss<<"i="<<setw(4)<<i<<" x "<<x.row(i);
                cout<<setw(40)<<oss.str();
            }
            cout<<" proj "<<projections.col(i).transpose()<<endl;
        }
    }
    // projections.rows() [ nExamples x nProj ];
    std::vector<boost::dynamic_bitset<>> active;
    projectionsToBitsets( projections, s, active );
    return active;
}

/** xform {l,u} of a soln into permutations for fast class [lookup and] \b scoring. */
struct MCprojector {
    MCprojector( MCsoln const& s );
    std::vector<SimpleProjectionScores> score( DenseM const& p );
private:
    /// \name work areas, as [ nProj x nClass ] row-wise matrices for speed
    //@{
    // XXX CHANGE these to calc as [ nProj x nClass ] matrices
    //     (ie. calc ONCE for all examples and projections)
    DenseM lb;
    DenseM ub;
    DenseM mid;
    DenseM lSlope;      ///< 1.0/(mid-lb)
    DenseM uSlope;      ///< 1.0/(ub-mid)
    //@}
    VectorXd scDbl;       ///< for ONE example [ nClass ]
};

#if 0
/** For each example, develop \f$min_p(centrality_c(proj[p],l_p,u_p,...))\f$ over all
 * projections p in \c proj and classes c, for some [simple] \c centrality score.
 *
 * \p proj      [ nExamples x nProj ] row-wise projection coeffs of each example
 */
std::vector<SimpleProjectionScores> score( DenseM const& proj, MCsoln  );
#endif

void projectionsToScores( DenseM const& projections,
                          MCsoln const& s,
                          uint32_t const targetSize,
                          std::vector<SimpleProjectionScores>& sps );

template<typename EigenType>
std::vector<SimpleProjectionScores> project( EigenType const& x,
                                             MCsoln const& s,
                                             uint32_t const targetSize )
{
    assert( x.cols() == s.weights_avg().rows() );
    DenseM const projections = (x * s.weights_avg).transpose();
    std::vector<SimpleProjectionScores> sps;
    projectionsToScores( projections, s, targetSize, sps );
    return sps;
}
#endif // MCPREDICT_HH
