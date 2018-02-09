#include "find_w_detail.h"
#include "boost/iterator/counting_iterator.hpp"
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <cmath>

#include "constants.h"

#define __restricted /* __restricted seems to be an error */

/** 0 --> Alex's original dev-branch version
 * 1 --> my version */
#define OPTIMIZE_LU_VERSION 1

using namespace std;

//*****************************************
// function used by optimizeLU
// grad are stored in order of the ranked classes
// to minimize cash misses and false sharing

/** set sorted class_order values of labels for training example 'idx' */
static inline void sortedClasses( std::vector<int> & classes, std::vector<int> const& class_order,
                           SparseMb const& y, size_t const idx )
{
    classes.resize(0);
    for (SparseMb::InnerIterator it(y,idx); it; ++it){
        //assert(it.value()); // if (it.value())
        classes.push_back(class_order[it.col()]);
    }
    std::sort(classes.begin(),classes.end());
}

//*****************************************
// function used by optimizeLU 
// grad are stored in order of the ranked classes
// to minimize cash misses and false sharing 

static inline void getBoundGrad (VectorXd& __restricted grad, VectorXd& __restricted bound,
                          const size_t idx, const size_t allproj_idx,
                          const std::vector<int>& __restricted sorted_class,
                          const int sc_start, const int sc_end,
                          const std::vector<int>& __restricted classes,
                          const double start_update, const double other_weight,
                          const VectorXd& __restricted allproj,
			  const bool none_filtered, const boolmatrix& __restricted filtered)

{
  std::vector<int>::const_iterator class_iter = std::lower_bound(classes.begin(), classes.end(), sc_start);  
  double update = start_update + (class_iter - classes.begin())*other_weight;
// #pragma omp critical
//   {
//     cout << "1  " << idx << "   " << allproj_idx << "   " << sc_start << "   " << sc_end << "    " << class_iter - classes.begin() << "  " << update << "   " << other_weight << endl;
//   }
  for (int sc = sc_start; sc < sc_end; sc++)
    {
      if (class_iter != classes.end() && sc == *class_iter)
	{
	  // example is of this class
	  update += other_weight;
	  class_iter++;
	  continue;
	}		  
      const int cp = sorted_class[sc];
      if (grad.coeff(sc) >= 0 && (none_filtered || !(filtered.get(idx,cp))))
	{			      
	  grad.coeffRef(sc) -= update;
	  if (grad.coeff(sc) < 0)
	    {
	      bound.coeffRef(cp) = allproj.coeff(allproj_idx);
	    }
	}
    }

}

/** get optimal {l,u} bounds given projection and class order.
 * Computationally expensive, so it should be done sparingly.
 */
void optimizeLU(VectorXd& l, VectorXd& u,
		const VectorXd& projection, const SparseMb& y,
		const vector<int>& class_order, const vector<int>& sorted_class,
		const VectorXd& wc, const VectorXi& nclasses,
		const boolmatrix& filtered,
		const double C1, const double C2,
		const param_struct& params,
		bool print)
{
    // -------------- various threading options -------------
    // Good for MCTHREADS==0: UPFRONT=0 && BOUNDGRAD_THREAD=1 is good
    // For MCTHREADS==1:     BOUNDGRAD_THREAD=0, BOUNDGRAD_THREAD=0 
#define BOUNDGRAD_THREAD 1 // 1=good

    // -------------- and sanity checks
#if MCTHREADS && defined(_OPENMP)
    if( omp_in_parallel() )
        throw runtime_error(" ERROR: please don't call optimizeLU from and omp parallel section");
#endif
    static bool check_once=false;
    if( !check_once ){
        if( params.remove_class_constraints ){
            cout<<" WARNING: optimizelu gradient massage via wc.coeff may not work correctly"<<endl;
        }
        size_t nErase=0U;
#if MCTHREADS && defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for(size_t o=0U; o<y.outerSize(); ++o){
            for(SparseMb::InnerIterator it(y,o); it; ++it) {
                if( it.value()==false ){
                    ++nErase;
                }
            }
        }
        if( nErase )
            throw runtime_error(" ERROR: SparseMb 'y' should have only 'true' entries");
        if( ! y.isCompressed() )
            throw runtime_error(" ERROR: expect SparseMb 'y' to be compressed sparse");
        check_once=true;
    }
    // ------------------------------------------------------
    size_t const n = projection.size();
    size_t const noClasses = y.cols();
    bool const none_filtered = filtered.count()==0;
    VectorXd allproj(2*n);
    allproj << (projection.array() - 1), (projection.array() + 1);
    std::vector<size_t> indices(allproj.size());
    sort_index(allproj, indices);

    // yes, co[sc[c]] == c
    // initialize the gradients for u and l 
    VectorXd gradu(noClasses); // by ranked classes, to minimize cache misses/false sharing
    VectorXd gradl(noClasses); // by ranked classes, to minimize cache misses/false sharing
#if MCTHREADS && defined(_OPENMP)
    size_t const ck = std::max(size_t{256U},size_t{noClasses/omp_get_max_threads()});
    //#pragma omp parallel for simd schedule(static,ck)
#pragma omp parallel for schedule(static,ck)
#endif
    for (size_t sc = 0; sc <noClasses; ++sc) {
      int const c = sorted_class[sc];
      double const classweight = wc.coeff(c); // incorrect if remove_class_constraints because true classes can be filtered. Need to fix this by recalculating wc after filtering
      if (classweight <= 0.0) { // no examples of this class
	// classes with no weight (wc[]) get {l,u} bounds set to high/low values.
	u.coeffRef(c) = boost::numeric::bounds<double>::lowest()/10; // the /10 is needed because octave has trouble reading back ascii files that are written with the highest/lowest limits
	l.coeffRef(c) = boost::numeric::bounds<double>::highest()/10; // the /10 is needed because octave has trouble reading back ascii files that are written with the highest/lowest limits;
	gradu.coeffRef(sc) = -1.0;
	gradl.coeffRef(sc) = -1.0;
      } else {
	gradu.coeffRef(sc) = C1*classweight;
	gradl.coeffRef(sc) = C1*classweight;
      }
    }


    
#if BOUNDGRAD_THREAD && MCTHREADS && defined(_OPENMP)
    int const max_n_chunks = omp_get_max_threads();

    int const min_chunk_size = 10000;     // need a test case to set this value XXX
    //int min_chunk_size = 100;
    //if( noClasses/max_n_chunks > min_chunk_size ) min_chunk_size = noClasses/max_n_chunks;
    //if( min_chunk_size > 10000 ) min_chunk_size = 10000;
#endif

#if MCTHREADS
#pragma omp parallel default(none) shared(l, u, allproj, indices, filtered, y, class_order, sorted_class, wc, nclasses, params, gradu, gradl)
#endif
    {
#if MCTHREADS
#pragma omp single nowait
#endif
      { // calculate the optimal value for upper bounds -- iterate from beginning
        std::vector<int> classes;
        classes.reserve(nclasses.maxCoeff());
        for (std::vector<size_t>::const_iterator i = indices.begin(); i != indices.end(); ++i) {
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n) { plus = true; idx -= n; }
	  
	  if (plus) { // only the upper bounds of the classes of this example are affected
	    double const class_weight = (params.ml_wt_class_by_nclasses
					 ? C1 / nclasses.coeff(idx): C1);
	    for (SparseMb::InnerIterator it(y,idx); it; ++it) {
	      //assert( it.value() ); if (it.value()) ...
	      int const cs = it.col();                // raw [unsorted] class
	      int const sc = class_order[cs];         // sorted class number
	      if( gradu.coeff(sc) >= 0 && (gradu.coeffRef(sc) -= class_weight) <  1e-10){
		u.coeffRef(cs) = allproj.coeff(*i);
	      }
	    }
	  }else{
	    // only the classes ranked lower than the classes of this example are affected
	    sortedClasses( /*OUT*/ classes, /*IN*/ class_order, y, idx );
	    if (classes.size() == 0) continue;
	    if (classes.back() == 0) continue;
	    
	    double const other_weight = (params.ml_wt_by_nclasses
					 ? C2 / nclasses.coeff(idx): C2);
	    // how many classes of the curent instance should be ranked higher 
	    //  times the weight of each
	    //  if each class has its own weight will need to
	    //  be calculated below (or have it precomputed for each example 
	    //  as a corresponding wclasses to nclasses to be wclasses the same as wc 
	    //  corresponds to nc
	    double const right_update = other_weight * nclasses.coeff(idx);
#if BOUNDGRAD_THREAD && MCTHREADS && defined(_OPENMP)
	    // make sure there is enough work to do to paralelize this
	    int n_chunks = classes.back()/min_chunk_size + 1;
	    n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
	    if( n_chunks > 1 ){
	      int chunk_size = classes.back()/n_chunks;
	      int remaining = classes.back()%n_chunks;
	      for (int chunk=0; chunk < n_chunks; ++chunk)
		{
#pragma omp task default(shared) firstprivate(chunk) shared(gradu, u, idx, i, sorted_class, classes, allproj, filtered, chunk_size, remaining)
		  {
		    int sc_start = chunk*chunk_size + (chunk<remaining?chunk:remaining);
		    int sc_incr = chunk_size + (chunk<remaining);
		    getBoundGrad(gradu, u, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, right_update, -other_weight, allproj, none_filtered, filtered);
		  }
#pragma omp taskwait // need to wait for all tasks to finish before moving to the next example
		}
	    }else{
	      getBoundGrad(gradu, u, idx, *i, sorted_class, 0,classes.back(),classes,right_update,-other_weight,allproj,none_filtered,filtered);
	    }
#else // not _OPENMP
	    getBoundGrad(gradu, u, idx, *i, sorted_class, 0,classes.back(),classes,right_update,-other_weight,allproj,none_filtered,filtered);
#endif // _OPENMP	   
	  }
        }
      }
      
      
#if MCTHREADS
#pragma omp single nowait
#endif
      { // calculate the optimal value for lower bounds -- iterate from end
        std::vector<int> classes;
        classes.reserve(nclasses.maxCoeff());
        for (std::vector<size_t>::const_reverse_iterator i = indices.rbegin(); i != indices.rend(); i++) {
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n) { plus = true; idx -= n; }
	  
	  if (!plus){ // only the lower bounds of the classes of this example are affected
	    double const class_weight = (params.ml_wt_class_by_nclasses
					 ? C1 / nclasses.coeff(idx): C1);
	    for (SparseMb::InnerIterator it(y,idx); it; ++it) {
	      //assert( it.value() ); //if (it.value())
	      int cs = it.col();
	      int sc = class_order[cs];
	      if (gradl.coeff(sc) >= 0 && (gradl.coeffRef(sc) -= class_weight) < 1e-10){
		l.coeffRef(cs) = allproj.coeff(*i);
	      }
	    }
	  }else{
	    // only the classes ranked higher than the classes of this example are affected
	    // calling y.coeff is expensive so get the classes in the ranked order here
	    sortedClasses( /*OUT*/ classes, /*IN*/ class_order, y, idx );
	    if(classes.size() == 0U) continue;
	    // if a class has lower rank than the lowest rank class of this example
	    // it's lower bound will not be influenced by this example
	    int n_active = noClasses - classes.front() - 1;
	    if (n_active <= 0)
	      continue;
	    
	    double const other_weight = (params.ml_wt_by_nclasses
					 ? C2 / nclasses.coeff(idx): C2);
#if BOUNDGRAD_THREAD && MCTHREADS && defined(_OPENMP)
	    // make sure there is enough work to do to paralelize this
	    int n_chunks = n_active/min_chunk_size + 1;
	    n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
	    if( n_chunks > 1 ){
	      int chunk_size = n_active/n_chunks;
	      int remaining = n_active%n_chunks;
	      for (int chunk=0; chunk < n_chunks; ++chunk)
		{
#pragma omp task default(shared) firstprivate(chunk) shared(gradl, l, idx, i, sorted_class, classes, allproj, filtered, chunk_size, remaining)
		  {
		    int sc_start = classes.front() + 1 + chunk*chunk_size + (chunk<remaining?chunk:remaining);
		    int sc_incr = chunk_size + (chunk<remaining);
		    getBoundGrad(gradl, l, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, 0.0, other_weight, allproj, none_filtered, filtered);
		  }
#pragma omp taskwait  // need to wait for all tasks to finish before moving to the next example
		}
	    }else{
	      getBoundGrad(gradl, l, idx, *i, sorted_class, classes.front() + 1,noClasses,classes, 0.0, other_weight, allproj, none_filtered, filtered);
	    }
#else // not _OPENMP
	    getBoundGrad(gradl, l, idx, *i, sorted_class, classes.front() + 1,noClasses,classes, 0.0, other_weight, allproj, none_filtered, filtered);
#endif // _OPENMP
	  }
        }
      } // omp single
    } // omp parallel
   
}

