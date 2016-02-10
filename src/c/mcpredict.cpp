
#include "mcpredict.h"
#include "filter.h"
#include <boost/dynamic_bitset.hpp>
#include <iostream>

using namespace std;

void projectionsToBitsets( DenseM const& projections,
                           MCsoln const& s,
                           std::vector<boost::dynamic_bitset<>>& active )
{
    int const verbose=0;
    size_t const nExamples = projections.cols();        // TRANSPOSE of x*w, for efficiency
    size_t const nProj = projections.rows();
    size_t const nClass = s.lower_bounds_avg.rows();
    assert( (int)nClass == s.upper_bounds_avg.rows() );
    assert( (int)nClass == s.lower_bounds_avg.rows() );
    assert( (int)nProj == s.upper_bounds_avg.cols() );
    assert( (int)nProj == s.lower_bounds_avg.cols() );

    //active = new ActiveDataSet(nExamples);
    if( active.size() > nExamples ){
        active.resize(nExamples);
    }
    for(size_t i=0U; i<active.size(); ++i){
        active[i].clear();
        active[i].resize(nClass,true);
    }
    if( active.size() < nExamples ){
        if( active.size() == 0U ){
            active.emplace_back( boost::dynamic_bitset<>() );
            active.back().resize(nClass,true);
        }
        assert( active.size() > 0U );
        size_t bk = active.size() - 1U;
        while( active.size() < nExamples )
            active.emplace_back( active[bk] );  // can copy-construct all-true as a copy in 1 step
    }
    assert( active.size() == nExamples );
#if 1 // slow!
    for( uint32_t e=0U; e<nExamples; ++e ){
        assert( active[e].size() == nExamples );
        assert( active[e].count() == nExamples );
    }
#endif

    //VectorXd proj;
    VectorXd l;
    VectorXd u;
    // TODO if ! nProj >> nExamples, provide a faster impl ???
    for (int i = 0; i < nProj; i++)
    {
        //proj = projections.row(i);
        l = s.lower_bounds_avg.col(i);
        u = s.upper_bounds_avg.col(i);
        if (verbose) cout << "Init filter" << endl;
        Filter f( l, u );
        if (verbose) cout << "Update filter, projection " << i << endl;
#if MCTHREADS
#pragma omp parallel for shared(f,projections,active,i)
#endif
        for(size_t j=0; j < nExamples; ++j)
        {
            active[j] &= *(f.filter(projections.coeff(i,j)));
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
