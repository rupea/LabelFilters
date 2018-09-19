#include "mcsolver.h"
#include "mcsolver.hh"
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <algorithm>    // min,max
#include <assert.h>

using namespace std;


// Complete some of the class declarations before instantiating MCsolver
MCsolver::MCsolver()
  : MCsoln(),objective_val()
{
}

MCsolver::MCsolver(MCsoln const& soln)
  : MCsoln(soln),objective_val()
{}

MCsolver::~MCsolver()
{
  //cout<<" ~MCsolver--TODO: where to write the MCsoln ?"<<endl;
}


// Explicitly instantiate MCsolver into the library

template
void MCsolver::solve( DenseM const& x, SparseMb const& y, param_struct const& params_arg );
template
void MCsolver::solve( SparseM const& x, SparseMb const& y, param_struct const& params_arg );
template
void MCsolver::solve( ExtConstSparseM const& x, SparseMb const& y, param_struct const& params_arg );


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
  , ok_sortlu(false)
  , ok_lu_avg(false)
  , ok_sortlu_avg(false)
    
  , m_l( nClass )
  , m_u( nClass )
  , m_sortlu( nClass*2U )
    
  , m_l_acc( nClass )
  , m_u_acc( nClass )
  , m_sortlu_acc( nClass*2U )
  , nAccSortlu(0U)

  , m_l_avg( nClass )
  , m_u_avg( nClass )
  , m_sortlu_avg( nClass*2U )
    
{
  reset();            // just in case Eigen does not zero them initially
}

void MCpermState::reset()
{
  // starting from l = u = zero. l and u and sortlu are ok. 
  ok_lu = true;
  ok_sortlu = true;
  m_l.setZero();
  m_u.setZero();
  m_sortlu.setZero();
  reset_acc();
  
  ok_lu_avg = false;
  m_l_avg.setZero();
  m_u_avg.setZero();
  m_sortlu_avg.setZero();
  ok_sortlu_avg = false;

}
void MCpermState::init( /* inputs: */ VectorXd const& projection, SparseMb const& y,
			const param_struct& params, boolmatrix const& filtered)
{
  size_t const nClasses = y.cols();
  size_t const nEx      = y.rows();
  assert( m_l.size() == (int)nClasses );
  assert( m_u.size() == (int)nClasses );
  assert( projection.size() == (int)nEx );
  m_l.setConstant(  0.1 * boost::numeric::bounds<double>::highest() );
  m_u.setConstant(  0.1 * boost::numeric::bounds<double>::lowest() );
  for (size_t i=0; i<nEx; ++i) {
    for (SparseMb::InnerIterator it(y,i); it; ++it) {
      if (it.value()) {
	size_t const c = it.col();
	assert( c < nClasses );
	if (!params.remove_class_constraints || !(filtered.get(i,c)))
	  {	    
	    double const pr = projection.coeff(i);
	    m_l.coeffRef(c) = min(pr, m_l.coeff(c));
	    m_u.coeffRef(c) = max(pr, m_u.coeff(c));
	  }
      }
    }
  }
  ok_lu = true;
  ok_sortlu = false;
  reset_acc();
  ok_sortlu_avg = false;
  ok_lu_avg = false;
}

void MCpermState::optimizeLU( VectorXd const& projection, SparseMb const& y, VectorXd const& wc,
                              VectorXi const& nclasses, 
			      VectorXd const& inside_weight, VectorXd const& outside_weight,
			      boolmatrix const& filtered,
                              double const C1, double const C2,
                              param_struct const& params )
{
  assert(nAccSortlu == 0); // we either use optimizeLU or we accumulate gradients. Not both. 
  mcsolver_detail::optimizeLU( m_l, m_u, // <--- outputs
			       projection, y, rev()/*class_order*/, perm()/*sorted_class*/, wc,
			       nclasses, inside_weight, outside_weight, filtered, C1, C2, params );
  ok_lu = true;
  ok_sortlu = false;
  reset_acc();
  ok_sortlu_avg = false;
}

void MCpermState::optimizeLU_avg( VectorXd const& projection_avg, SparseMb const& y, VectorXd const& wc,
                                  VectorXi const& nclasses, 
				  VectorXd const& inside_weight, VectorXd const& outside_weight,
				  boolmatrix const& filtered,
                                  double const C1, double const C2,
                                  param_struct const& params )
{
  assert(nAccSortlu == 0); // we either use optimizeLU or we accumulate gradients. Not both. 
  mcsolver_detail::optimizeLU( m_l_avg, m_u_avg, // <--- outputs
			       projection_avg, y, rev()/*class_order*/, perm()/*ssorted_class*/, wc,
			       nclasses, inside_weight, outside_weight, filtered, C1, C2, params );
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
    , idx_locks         ( params.update_type==MINIBATCH_SGD? new MutexType[idx_chunks]: nullptr )
    , sc_locks          ( params.update_type==MINIBATCH_SGD? new MutexType[sc_chunks]: nullptr )
{}
MCupdateChunking::~MCupdateChunking(){
  if (idx_locks) {delete[] idx_locks; const_cast<MutexType*&>(idx_locks) = nullptr;}
  if (sc_locks) {delete[]  sc_locks; const_cast<MutexType*&>( sc_locks) = nullptr;}
}


