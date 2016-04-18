
#include "mcsolver.h"
#include "find_w_detail.h"      // ::optimizeLU
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <omp.h>
#include <iostream>
#include <algorithm>    // min,max
#include <assert.h>

using namespace std;

// ... MCxyData magic headers (simplify I/O)
std::array<char,4> MCxyData::magic_xSparse = {0,'X','s','8'}; // or 4 for floats
std::array<char,4> MCxyData::magic_xDense  = {0,'X','d','8'}; // or 4 for floats
// x text mode not supported so far.
std::array<char,4> MCxyData::magic_yBin    = {0,'Y','s','b'};
// y text mode readable but has no magic.
// ...
/** Set solution sizes for nProj projections.
 * Typically used to chop un-needed projections.
 * ?? should it also handle increasing number
 *    of projections (and zero-initializing all) ??
 */
void MCsolver::chopProjections(size_t const nProj){
    cerr<<" chopProjections("<<nProj<<")"<<endl;
    if(weights.cols() > nProj){
        cerr<<"Reducing weights from "<<weights.cols()<<" to "<<nProj<<" projections"<<endl;
        weights.conservativeResize(/*d*/weights.rows(), nProj);
        lower_bounds.conservativeResize(/*nClass*/lower_bounds.rows(), nProj);
        upper_bounds.conservativeResize(/*nClass*/upper_bounds.rows(), nProj);
    }
    if(weights_avg.cols() > nProj){
        cerr<<"Reducing weights_avg from "<<weights_avg.cols()<<" to "<<nProj<<" projections"<<endl;
        weights_avg.conservativeResize(/*d*/weights_avg.rows(), nProj);
        lower_bounds_avg.conservativeResize(/*nClass*/lower_bounds_avg.rows(), nProj);
        upper_bounds_avg.conservativeResize(/*nClass*/upper_bounds_avg.rows(), nProj);
    }
}

    MCpermState::MCpermState( size_t nClass )
    : Perm(nClass)
    , ok_lu(false)
    , ok_lu_avg(false)
    , ok_sortlu(false)
      , ok_sortlu_avg(true)

    , l( nClass )
    , u( nClass )
    , sortlu( nClass*2U )

    , sortlu_avg( nClass*2U )
    , nAccSortlu_avg(0U)

    , l_avg( nClass )
    , u_avg( nClass )
{
    cout<<" +MCpermState(nClass="<<nClass<<")"<<endl;
    reset();            // just in case Eigen does not zero them initially
}

void MCpermState::reset()
{

    l.setZero();
    u.setZero();
    sortlu.setZero();
    ok_lu = false;
    ok_lu_avg = false;

    sortlu_avg.setZero();
    ok_sortlu_avg = true;       // sortlu_avg is an ACUUMULATOR - setZero is a valid initial state
    nAccSortlu_avg = 0U;            // but sortlu_avg becomes significant only when this is nonzero

    l_avg.setZero();
    u_avg.setZero();
    ok_sortlu = false;
}
void MCpermState::init( /* inputs: */ VectorXd const& projection, SparseMb const& y, VectorXi const& nc )
{
    size_t const nClasses = y.cols();
    size_t const nEx      = y.rows();
    cout<<" MCpermState::init(projection["<<projection.size()<<"],y,nc) nClasses="<<nClasses<<")"<<endl;
    //l.conservativeResize( nClasses );
    //u.conservativeResize( nClasses );
    assert( l.size() == (int)nClasses );
    assert( u.size() == (int)nClasses );
    assert( projection.size() == (int)nEx );
    l.setConstant(  0.1 * boost::numeric::bounds<double>::highest() );
    u.setConstant(  0.1 * boost::numeric::bounds<double>::lowest() );
    for (size_t i=0; i<nEx; ++i) {
        for (SparseMb::InnerIterator it(y,i); it; ++it) {
            if (it.value()) {
                size_t const c = it.col();
                assert( c < nClasses );
                double const pr = projection.coeff(i);
                l.coeffRef(c) = min(pr, l.coeff(c));
                u.coeffRef(c) = max(pr, u.coeff(c));
            }
        }
    }
//#ifndef NDEBUG
//    for(size_t c=0U; c<nClasses; ++c){
//        assert( l.coeff(c) <= u.coeff(c) );
//    }
//#endif
    ok_lu = true;
    ok_sortlu = false;

    sortlu_avg.setZero();
    ok_sortlu_avg = true;
    nAccSortlu_avg = 0U;

    ok_lu_avg = false;
}

void MCpermState::optimizeLU( VectorXd const& projection, SparseMb const& y, VectorXd const& wc,
                              VectorXi const& nclasses, boolmatrix const& filtered,
                              double const C1, double const C2,
                              param_struct const& params, bool print )
{
//#ifndef NDEBUG
//        assert( l.size() == u.size() );
//        for(size_t c=0U; c<l.size(); ++c){
//            assert( l.coeff(c) <= u.coeff(c) );
//        }
//#endif
    ::optimizeLU( l, u, // <--- outputs
                  projection, y, rev/*class_order*/, perm/*sorted_class*/, wc,
                  nclasses, filtered, C1, C2, params, print );
//#ifndef NDEBUG
//        assert( l.size() == u.size() );
//        for(size_t c=0U; c<l.size(); ++c){
//            assert( l.coeff(c) <= u.coeff(c) );
//        }
//#endif
    ok_lu = true;
    ok_sortlu = false;
}
void MCpermState::optimizeLU_avg( VectorXd const& projection_avg, SparseMb const& y, VectorXd const& wc,
                                  VectorXi const& nclasses, boolmatrix const& filtered,
                                  double const C1, double const C2,
                                  param_struct const& params, bool print )
{
    ::optimizeLU( l_avg, u_avg, // <--- outputs
                  projection_avg, y, rev/*class_order*/, perm/*ssorted_class*/, wc,
                  nclasses, filtered, C1, C2, params, print );
    ok_lu_avg = true;
    // NO EFFECT on sortlu_avg, which is an ACCUMULATOR-OF-sortlu
    //ok_sortlu_avg = true;
}

int MCsolver::getNthreads( param_struct const& params ) const
{
    int nThreads;
#ifdef _OPENMP
    if (params.num_threads < 0)
        nThreads = omp_get_num_procs();   // use # of CPUs
    else if (params.num_threads == 0)
        nThreads = omp_get_max_threads(); // use OMP_NUM_THREADS
    else
        nThreads = params.num_threads;
    omp_set_num_threads( nThreads );
    Eigen::initParallel();
    cout<<" solve_ with _OPENMP and params.num_threads set to "<<params.num_threads
        <<", nThreads is "<<nThreads<<", and omp_max_threads is now "<<omp_get_max_threads()<<endl;
    // NOTE: omp_get_num_threads==1 because we are not in an omp section
#else
    nThreads = 1;
    cout<<" no _OPENMP ";
#endif
    return nThreads;
}

MCupdateChunking::MCupdateChunking( size_t const nTrain, size_t const nClass,
                                           size_t const nThreads, param_struct const& params )
    : batch_size( [nTrain,&params]{
                  size_t const bs = min(max( size_t{1}, size_t{params.batch_size} ), nTrain);
#ifndef NDEBUG
                  if (params.update_type == SAFE_SGD) assert(bs == 1U);
#endif
                  return bs;
                  }() )
    , sc_chunks         ( nThreads )
    , sc_chunk_size     ( (params.class_samples?params.class_samples:nClass) / sc_chunks )
    , sc_remaining      ( (params.class_samples?params.class_samples:nClass) % sc_chunks )
    , idx_chunks        ( nThreads )
    , idx_chunk_size    ( batch_size / idx_chunks )
    , idx_remaining     ( batch_size % idx_chunks )
    , idx_locks         ( new MutexType[idx_chunks] )
    , sc_locks          ( new MutexType[idx_chunks] )
    //, idx_locks         ( params.update_type==MINIBATCH_SGD? new MutexType[idx_chunks]: nullptr )
    //, sc_locks          ( params.update_type==MINIBATCH_SGD? new MutexType[sc_chunks]: nullptr )
{}
MCupdateChunking::~MCupdateChunking(){
    delete[] idx_locks; const_cast<MutexType*&>(idx_locks) = nullptr;
    delete[]  sc_locks; const_cast<MutexType*&>( sc_locks) = nullptr;
}


