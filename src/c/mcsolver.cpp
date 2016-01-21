
#include "mcsolver.h"
#include "find_w_detail.h"      // ::optimizeLU
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <omp.h>
#include <iostream>
#include <algorithm>    // min,max
#include <assert.h>

using namespace std;

    MCpermState::MCpermState( size_t nClass )
    : Perm(nClass)
    , ok_lu(false)
    , ok_lu_avg(false)
    , ok_sortlu(false)
      , ok_sortlu_avg(true)

    , l( nClass )
    , u( nClass )
, sortlu( nClass*2U )

    , l_avg( nClass )
    , u_avg( nClass )
, sortlu_avg( nClass*2U )
{
    reset();            // just in case Eigen does not zero them initially
}

void MCpermState::reset()
{

    l.setZero();
    u.setZero();
    sortlu.setZero();
    ok_lu = false;
    ok_lu_avg = false;

    l_avg.setZero();
    u_avg.setZero();
    sortlu_avg.setZero();
    ok_sortlu = false;
    ok_sortlu_avg = true;       // sortlu_avg is an ACUUMULATOR - it begins at zero, which is a valid state.
}
void MCpermState::init( /* inputs: */ VectorXd const& projection, SparseMb const& y, VectorXi const& nc )
{
    l.setConstant(  0.1 * boost::numeric::bounds<double>::highest() );
    u.setConstant(  0.1 * boost::numeric::bounds<double>::lowest() );
    size_t const nClasses = y.rows();
    // XXX if we iterate over CLASSES, then loop can be parallelized, maybe
    for (size_t i=0; i<nClasses; ++i) {
        for (SparseMb::InnerIterator it(y,i); it; ++it) {
            if (it.value()) {
                size_t const c = it.col();
                double const pr = projection.coeff(i);
                l.coeffRef(c) = min(pr, l.coeff(c));
                u.coeffRef(c) = max(pr, u.coeff(c));
            }
        }
    }
    ok_lu = true;
    ok_sortlu = false;

    sortlu_avg.setZero();
    ok_sortlu_avg = true;
    ok_lu_avg = false;
}

void MCpermState::optimizeLU( VectorXd const& projection, SparseMb const& y, VectorXd const& wc,
                              VectorXi const& nclasses, boolmatrix const& filtered,
                              double const C1, double const C2,
                              param_struct const& params, bool print )
{
    ::optimizeLU( l, u, projection, y, rev/*class_order*/, perm/*ssorted_class*/, wc,
                  nclasses, filtered, C1, C2, params, print );
    ok_lu = true;
    ok_sortlu = false;
}
void MCpermState::optimizeLU_avg( VectorXd const& projection_avg, SparseMb const& y, VectorXd const& wc,
                                  VectorXi const& nclasses, boolmatrix const& filtered,
                                  double const C1, double const C2,
                                  param_struct const& params, bool print )
{
    ::optimizeLU( l_avg, u_avg, projection_avg, y, rev/*class_order*/, perm/*ssorted_class*/, wc,
                  nclasses, filtered, C1, C2, params, print );
    ok_lu_avg = true;
    ok_sortlu_avg = false;
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
    cout<<" solve_ with _OPENMP and params.num_threads set to "<<params.num_threads
        <<", nThreads is "<<nThreads<<", and omp_max_threads is now "<<omp_get_max_threads()<<endl;
    // NOTE: omp_get_num_threads==1 because we are not in an omp section
#else
    nThreads = 1;
    cout<<" no _OPENMP ";
#endif
    return nThreads;
}

#if MCUC
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
#endif


