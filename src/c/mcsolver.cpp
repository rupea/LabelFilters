#include "mcsolver.h"
#include "mcsolver.hh"
#include "mcxydata.h"

#include "find_w_detail.h"      // ::optimizeLU
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <omp.h>
#include <iostream>
#include <fstream>
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



// Complete some of the class declarations before instantiating MCsolver
MCsolver::MCsolver()
  : MCsoln(),objective_val()
    // private "solve" variables here TODO
{
  // if( solnfile ){
  //   ifstream ifs(solnfile);
  //   if( ifs.good() ) try{
  // 	cout<<" reading "<<solnfile<<endl;
  // 	this->read( ifs );
  // 	this->pretty( cout );
  // 	cout<<" reading "<<solnfile<<" DONE"<<endl;
  //     }catch(std::exception const& e){
  // 	ostringstream err;
  // 	err<<"ERROR: unrecoverable error reading MCsoln from file "<<solnfile;
  // 	throw(runtime_error(err.str()));
  //     }
  // }
}

MCsolver::~MCsolver()
{
  //cout<<" ~MCsolver--TODO: where to write the MCsoln ?"<<endl;
}


// Explicitly instantiate MCsolver into the library

template
void MCsolver::solve( DenseM const& x, SparseMb const& y, param_struct const* const params_arg );
template
void MCsolver::solve( SparseM const& x, SparseMb const& y, param_struct const* const params_arg );
template
void MCsolver::solve( ExtConstSparseM const& x, SparseMb const& y, param_struct const* const params_arg );

// void MCsolver::trim( enum Trim const kp ){
//   if( kp == TRIM_LAST ){
//     // If have some 'last' data, swap {w,l,u} into {w,l,u}_avg
//     if( weights.size() != 0 ){
//       weights_avg.swap(weights);
//       lower_bounds_avg.swap(lower_bounds);
//       upper_bounds_avg.swap(upper_bounds);
//     }
//   }
//   // ** ALL ** the non-SHORT MCsoln memory is freed
//   // NOTE: in Eigen. resize always reallocates memory, so resize(0) WILL free memory.
//   objective_val_avg.resize(0);
//   weights.resize(0,0);
//   lower_bounds.resize(0,0);
//   upper_bounds.resize(0,0);
//   objective_val.resize(0);
// }


/** Set solution sizes for nProj projections.
 */
void MCsolver::setNProj(uint32_t const nProj, bool keep_weights, bool keep_LU)
{
  if (this->nProj > nProj && keep_weights)
    {
      cerr << "WARNING: only " << nProj 
	   << " projectsion are kept. The rest up to " << this->nProj 
	   << " are disregarded." << endl;
    }
  if (keep_weights)
    {
      weights.conservativeResize(d, nProj);
      for (uint32_t col= this->nProj; col < nProj; col++)
	{
	  weights.col(col).setZero();
	}
    }  
  else
    {
      weights.resize(d, nProj);
      weights.setZero();
    }      
  
  
  if (keep_LU)
    {
      lower_bounds.conservativeResize(nClass, nProj);
      upper_bounds.conservativeResize(nClass, nProj);
      for (uint32_t col= this->nProj; col < nProj; col++)
	{
	  lower_bounds.col(col).setZero();
	  upper_bounds.col(col).setZero();
	}
    }	
  else
    {
      lower_bounds.conservativeResize(nClass, nProj);
      upper_bounds.conservativeResize(nClass, nProj);
      lower_bounds.setZero();
      upper_bounds.setZero();
    }
  
  this->nProj = nProj;
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
    
  , sortlu_acc( nClass*2U )
  , nAccSortlu(0U)

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
  
  sortlu_acc.setZero();
  nAccSortlu = 0U;            // sortlu_avg becomes significant only when this is nonzero

  l_avg.setZero();
  u_avg.setZero();
  sortlu_avg.setZero();

  ok_lu = false;
  ok_lu_avg = false;
  ok_sortlu = false;
  ok_sortlu_avg = false;

}
void MCpermState::init( /* inputs: */ VectorXd const& projection, SparseMb const& y, VectorXi const& nc )
{
    size_t const nClasses = y.cols();
    size_t const nEx      = y.rows();
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
    nAccSortlu = 0U;

    ok_lu_avg = false;
}

void MCpermState::optimizeLU( VectorXd const& projection, SparseMb const& y, VectorXd const& wc,
                              VectorXi const& nclasses, boolmatrix const& filtered,
                              double const C1, double const C2,
                              param_struct const& params, bool print )
{
  ::optimizeLU( l, u, // <--- outputs
		projection, y, rev/*class_order*/, perm/*sorted_class*/, wc,
		nclasses, filtered, C1, C2, params, print );
  ok_lu = true;
  ok_sortlu = false;
  // reset accumulation since changes to lu do not come from a gradient step
  if (nAccSortlu > 0U)
    {
      nAccSortlu = 0U;
      sortlu_acc.setZero();
      ok_sortlu_avg = false;
    }    
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
    Eigen::initParallel();
    if (params.verbose >= 1) 
      {
	cout<<" solve_ with _OPENMP and params.num_threads set to "<<params.num_threads
	    <<", nThreads is "<<nThreads<<", and omp_max_threads is now "<<omp_get_max_threads()<<endl;
      }
    // NOTE: omp_get_num_threads==1 because we are not in an omp section
#else
    nThreads = 1;
    if (params.verbose >= 1)
      {
	cout<<" no _OPENMP ";
      }
#endif
    return nThreads;
}

MCupdateChunking::MCupdateChunking( size_t const nTrain, size_t const nClass,
                                           size_t const nThreads, param_struct const& params )
    : batch_size( [nTrain,&params]{
                  size_t bs = min(max( size_t{1}, size_t{params.batch_size} ), nTrain);
                  if (params.update_type == SAFE_SGD)
		    {
		      if ( bs != 1U ) 
			{
			  cerr << "WARNING: Setting batch size to 1 for SAFE_SGD updates" << endl;
			}
		      bs = 1;
		    }
		  return bs;
                  }() )
    , sc_chunks         ( nThreads )
    , sc_chunk_size     ( (params.class_samples?params.class_samples:nClass) / sc_chunks )
    , sc_remaining      ( (params.class_samples?params.class_samples:nClass) % sc_chunks )
    , idx_chunks        ( nThreads/sc_chunks )
    , idx_chunk_size    ( batch_size / idx_chunks )
    , idx_remaining     ( batch_size % idx_chunks )
    , idx_locks         ( new MutexType[idx_chunks] )
    , sc_locks          ( new MutexType[sc_chunks] )
    //, idx_locks         ( params.update_type==MINIBATCH_SGD? new MutexType[idx_chunks]: nullptr )
    //, sc_locks          ( params.update_type==MINIBATCH_SGD? new MutexType[sc_chunks]: nullptr )
{}
MCupdateChunking::~MCupdateChunking(){
    delete[] idx_locks; const_cast<MutexType*&>(idx_locks) = nullptr;
    delete[]  sc_locks; const_cast<MutexType*&>( sc_locks) = nullptr;
}


