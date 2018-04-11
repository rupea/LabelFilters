#include "mcpredict.hh"
#include "filter.h"
#include <iostream>

using namespace std;
using namespace Eigen;

void projectionsToBitsets( DenseM const& projections,
                           MCsoln const& s,
                           std::vector<Roaring>& active )
{
    int const verbose=0;
    if(verbose){ cout<<"projectionsToBitsets... MCsoln is:"<<endl; s.pretty(cout); }
    size_t const nExamples = projections.cols();        // TRANSPOSE of x*w, for efficiency
    size_t const nProj = projections.rows();
    size_t const nClass = s.lower_bounds.rows();
    assert( (int)nClass == s.upper_bounds.rows() );
    assert( (int)nClass == s.lower_bounds.rows() );
    assert( (int)nProj == s.upper_bounds.cols() );
    assert( (int)nProj == s.lower_bounds.cols() );

    //active = new ActiveDataSet(nExamples);
    // initialize with all labels active.   
    active.clear();
    active.reserve(nExamples);  
    Roaring full; //empty set
    full.flip(0,nClass); // full set
    full.setCopyOnWrite(false);
    for(size_t i=0U; i<nExamples; ++i){
      active.emplace_back(full);
    }
    assert( active.size() == nExamples );
    
    
    //VectorXd proj;
    VectorXd l;
    VectorXd u;
    // TODO if ! nProj >> nExamples, provide a faster impl ???
    for(size_t p=0U; p<nProj; ++p){
        //proj = projections.row(p);
        l = s.lower_bounds.col(p);
        u = s.upper_bounds.col(p);
        if (verbose) cout << "Init filter" << endl;
        Filter f( l, u );
        if (verbose) cout << "Update filter, projection " << p << endl;
#if MCTHREADS
#pragma omp parallel for shared(f,projections,active,p)
#endif
        for(size_t e=0U; e<nExamples; ++e){
	  Roaring const* dbitset = f.filter(projections.coeff(p,e));
            active[e] &= *dbitset;
            if(verbose && e==0){
                cout<<" projection value "<<projections.coeff(p,e)<<" Filter idx "<<f.idx(projections.coeff(p,e))<<endl;
                //cout<<"example 0: proj "<<p<<" dbitset="<<*dbitset<<" active="<<active[e]<<endl;
            }
        }
    }
}

void SimpleProjectionScores::sort()
{
    std::vector<uint32_t> perm(scores.size());
    std::iota( perm.begin(), perm.end(), uint32_t{0U} );
    std::sort( perm.begin(), perm.end(), [&](int const i, int const j)
               { return scores[i] > scores[j]; } );
    // Apply perm to both scores and classes
    auto asPerm = [&perm](int const i, int const j) { return perm[i] < perm[j]; };
    std::sort( scores.begin(), scores.end(), asPerm );
    std::sort( classes.begin(), classes.end(), asPerm );
}
    
void SimpleProjectionScores::prune(uint32_t const targetSize){
    if( size() > targetSize ){
        this->sort();
        assert( scores.back() >= minScore );
        assert( scores.back() == minScore );
        uint_least8_t m = scores[targetSize];
        if( m > minScore ){
            uint32_t newSz;
            for( newSz=targetSize; newSz < scores.size(); ++newSz )
                if( scores[newSz] < m )
                    break;
            assert( newSz < size() );
            --newSz;
            assert( scores[newSz] == m );
            minScore = m;
            scores.resize(newSz);
            classes.resize(newSz);
        }
    }
}
void SimpleProjectionScores::finalPrune(uint32_t const targetSize){
    if( size() > targetSize ){
        this->sort();
        assert( scores.back() >= minScore );
        assert( scores.back() == minScore );
        scores.resize(targetSize);
        scores.shrink_to_fit();
        classes.resize(targetSize);
        classes.shrink_to_fit();
        minScore = scores.back();
    }
}

SimpleProjectionScores::SimpleProjectionScores(uint32_t const reserve)
    : scores()
      , classes()
      , nNonZero(0U)
      , minScore(1U)
{
    scores.reserve(reserve);
    classes.reserve(reserve);
}

MCprojector::MCprojector( MCsoln const& s )
: lb()
    , ub()
    , mid()
    , lSlope()
    , uSlope()
    , scDbl()
{
    //uint32_t const nExamples = // unknown
    int const nProj = s.weights.cols();
    assert( s.lower_bounds.cols() == nProj );
    assert( s.upper_bounds.cols() == nProj );
    int const nClass = s.lower_bounds.rows();
    assert( s.upper_bounds.rows() == nClass );

    // work area setup:   These are all [ nProj x nClass ]
    lb = s.lower_bounds.transpose();
    ub = s.upper_bounds.transpose();
    mid = (lb+ub)*0.5;
    lSlope = (mid - lb).array().inverse();
    uSlope = (ub - mid).array().inverse();
    // except this one, which is a simple vector
    scDbl.resize( nClass );
}

std::vector<SimpleProjectionScores> MCprojector::score( DenseM const& proj )
{
    std::vector<SimpleProjectionScores> sps;
    uint32_t const nExamples = proj.cols();
    uint32_t const nProj = proj.rows();
    uint32_t const nClass = lb.cols();
    assert( (int)nProj == lb.rows() );
    assert( (int)nProj == ub.rows() );
    assert( (int)nClass == lb.cols() );
    assert( (int)nClass == ub.cols() );

    sps.reserve( nExamples );
    for(uint32_t e=0U; e<proj.rows(); ++e){             // for each example
        sps.emplace_back( SimpleProjectionScores(nClass) );
        auto & ee = sps.back();
        uint32_t p=0U;                                  // 1st projection
        for(uint32_t c=0U; c<nClass; ++c){
            // calc all scores, for the first one
            scDbl = (       proj.row(p) - lb.row(p)) .cwiseProduct( lSlope.row(p) ) /* lb-mid line */
                .cwiseMin( (ub.row(p) - proj.row(p)) .cwiseProduct( uSlope.row(p) ) /* mid-ub line */
                         );
            for(uint32_t c=0U; c<scDbl.size(); ++c){
                uint_least8_t score8 = static_cast<uint_least8_t>(std::max( scDbl.coeff(c), 0.0 ) * 255.999999);
                ee.add( c, score8 );
            }
        }
        //                                              // subsequent projections
        // ee is class-sorted, because we have neither pruned nor sorted
        assert( ee.classAscending() );
        // 1st calc ALL scores for all projections
        for(p=1U; p<nProj; ++p){                        // MERGE rest of scores

            // should TIME the two approaches and determine switch point for
            // ee.size() / nClass to switch between the two implementations
            if( ee.size() > nClass / 8U ){
                // if ee.size() is large, fastest to calculate ALL scores (nice SIMD)
                scDbl = (       proj.row(p) - lb.row(p)) .cwiseProduct( lSlope.row(p) ) /* lb-mid line */
                    .cwiseMin( (ub.row(p) - proj.row(p)) .cwiseProduct( uSlope.row(p) ) /* mid-ub line */
                             );
                //assert( scDbl < 1.0 );      // can be negative.
                // merge with existing class possibities in SimpleProjectionScores ee
                uint32_t eeIdx=0U;
                for(uint32_t c=0U; c<nClass; ){
                    while( eeIdx < ee.size() && ee.classes[eeIdx] < c ) ++eeIdx;
                    if( eeIdx >= ee.size() )
                        break;
                    while( c < ee.classes[eeIdx] ) ++c;
                    assert( ee.classes[eeIdx] == c );
                    assert( c < nClass );
                    // now have c allowed by eeIdx --- need to test c with proj[p]
                    // for this example e, projection p, class c
                    uint_least8_t const score8 = static_cast<uint_least8_t>(scDbl.coeff(c)*255.999999);
                    ee.merge( eeIdx, score8 );
                }// end for-classes c

            }else{ // otherwise calc scores "as needed", something like...

                // merge with existing class possibities in SimpleProjectionScores ee
                uint32_t eeIdx=0U;
                for(uint32_t c=0U; c<nClass; ){
                    while( eeIdx < ee.size() && ee.classes[eeIdx] < c ) ++eeIdx;
                    if( eeIdx >= ee.size() )
                        break;
                    while( c < ee.classes[eeIdx] ) ++c;
                    assert( ee.classes[eeIdx] == c );
                    assert( c < nClass );
                    // now have c allowed by eeIdx --- need to test c with proj[p]
                    uint_least8_t score8; // for this projection, class c, example e
                    {
                        double const c_proj = proj.coeff(e,p);
                        double const c_lo = (c_proj - lb.coeff(p,c)) * lSlope.coeff(p,c);      // 0.0 at lb, 1.0 at median
                        double const c_hi = (ub.coeff(p,c) - c_proj) * uSlope.coeff(p,c);      // 0.0 at ub, 1.0 at median
                        double const score = std::max( 0.0, std::min( c_lo, c_hi ));
                        assert( score <= 1.0 );
                        score8 = static_cast<uint_least8_t>(score * 255.999999);
                    }
                    ee.merge( eeIdx, score8 );
                }// end for-classes c
            }
        }//end for-projections p
    }//end for-examples e
    return sps;
}
void projectionsToScores( DenseM const& projections,
                          MCsoln const& s,
                          uint32_t const targetSize,
                          std::vector<SimpleProjectionScores>& sps )
{
    //int const verbose=0;
    //size_t const nExamples = projections.cols();        // TRANSPOSE of x*w, for efficiency
    size_t const nProj = projections.rows();
    size_t const nClass = s.lower_bounds.rows();
    assert( (int)nProj == s.upper_bounds.cols() );
    assert( (int)nProj == s.lower_bounds.cols() );
    assert( (int)nClass == s.upper_bounds.rows() );
    assert( (int)nClass == s.lower_bounds.rows() );
    MCprojector mcprojector( s );
    sps = mcprojector.score( projections );
}
