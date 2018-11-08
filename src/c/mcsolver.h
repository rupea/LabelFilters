/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCSOLVER_H
#define MCSOLVER_H
/** \file
 * helper classes to make MCsolver::solve more readable
 */
#include "mcsoln.h"

class Perm;             ///< an internal detail class -- no user access
class MCpermState;      ///< track which versions of {l,u}{,_avg} and sortedLU{,_avg} are official
class MCiterBools;      ///< t4_XXX bools for "Should we {params.XXX} at iteration MCsoln::t?"
class MCupdateState;    ///< utility constants/variables passed to update routine
namespace mcsolver_detail{
  struct MCupdate;
}
class boolmatrix; 

/** Provide a simpler API for solving.
 * - The long \c solve template is now in a 'detail' header,
 * - The library explicitly instiates for input data \c DenseM and SparseM.
 * - So no long headers are required to compile if you link with the library.
 *
 * - \b NOTE: eventually, one might move some of the other utility routines here ?
 */
class MCsolver : protected MCsoln
{
public:

    /**TODO: fix \b \c solve_optimization so proper resume can be done. */

  MCsolver();
  MCsolver(MCsoln const& soln);
  ~MCsolver();
  
  MCsoln       const& getSoln()  const {return *this;}
  MCsoln            & getSoln()        {return *this;}
  void read( std::istream& is )
  { MCsoln::read(is); }
  void write( std::ostream& os, enum Fmt fmt=BINARY) const
  { MCsoln::write(os,fmt); }
  
  /** solve for MCFilter's optimal multi-class discriminating projections.
   * \p x     training data, row-wise examples of dimension MCsoln::d
   * \p y     data classes, (should this be a vector? why a bool matrix?)
   * \p parms [opt.] parameters that overwrite MCsoln::parms
   * - if constructed with a solnfile, \throw if x/y dimensions are bad
   *   - o/w use x and y dims to initialize MCsoln data
   * - if \c p==nullptr, then use existing MCsoln::parms
   * \sa vectorToLabel to convert from y VectorXi of class numbers.
   * \internal
   * - Is there any benefit from y as vector of class numbers?
   *
   * - NOTE: EIGENTYPE \c DenseM and \c SparseM are provided by the default library.
   *   - Please only include \c find_w_detail.hh for \em strange 'x' types.
   */
  template< typename EIGENTYPE >
    void solve( EIGENTYPE const& x, SparseMb const& y, param_struct const& params_arg);
  
 private:
    Eigen::VectorXd objective_val;
    void setNProj(uint32_t const nProj, bool, bool);
    int getNthreads( param_struct const& params ) const;
};


/** Maintain a class-permutation and its inverse permutation */
class Perm {
private:
    friend class MCpermState; ///< allow access only via MCpermState
    Perm(size_t nClass) : m_perm(), m_rev()       ///< initialize to identity permutation
    {
        m_perm.reserve(nClass);
        m_rev.reserve(nClass);
        for(int i=0U, n=static_cast<int>(nClass); i<n; ++i){
            m_perm.push_back(i);
            m_rev.push_back(i);
        }
    }
    // \todo move back to int, when refactoring done
    std::vector<int> m_perm;  ///< forward permutation (ascending 'means') (old 'sorted_class')
    std::vector<int> m_rev;   ///< the reverse permutation (old 'class_order')
    /** produce new perm+rev according to ascending \c sortKey. */
    void rank( Eigen::VectorXd const& sortkey );
};

/** Some data is useful to maintain for post-processing operations
 * after MCsolver::solve has been called.
 */
class MCpermState : private Perm
{
public:
    friend class mcsolver_detail::MCupdate;      // perhaps temporarily
    MCpermState( size_t nClass );       ///< allocate 6 vectors of size nClass, 
    /** produce new perm+rev according to ascending \c sortKey. */
    void rank( Eigen::VectorXd const& sortkey ); // unperm, rerank, set flags

    /// \name Various ways to initialize \c l and \c u
    //@{
    /** starting from l = u = zero, return to freshly-constructed state. */
    void reset();

    /** init lower and upper bounds (lu) per class based on min-max projection values.
     * \p projection    dot-products of training examples with a particular projection axis
     * \p y             class labels for same training examples
     * \p nc            number training examples assigned to each class
     * - A quick substitute for the correct solution of \c optimizeLU.
     * \post \c ok_lu==true and \c ok_sortlu==false
     */
    void init( /* inputs: */ Eigen::VectorXd const& projection, SparseMb const& y,
	       const param_struct& params, boolmatrix const& filtered);

    /** If restarting from soln file, can also explicitly set initial state */
    template< typename EIGENTYPE1, typename EIGENTYPE2 >
    void set_lu( EIGENTYPE1 const& ll, EIGENTYPE2 const& uu  ){
      assert (m_l.size() == ll.size());
      assert ( ll.size() == uu.size() );
      m_l = ll;
      m_u = uu;
      ok_lu = true;
      ok_sortlu = false;
      reset_acc();
      ok_lu_avg = false;
      ok_sortlu_avg = false;
    }
    
    /** optimal settings for {l,u} */
    void optimizeLU( Eigen::VectorXd const& projection, SparseMb const& y, Eigen::VectorXd const& wc,
                     Eigen::VectorXi const& nclasses, 
		     Eigen::VectorXd const& inside_weight, Eigen::VectorXd const& outside_weight,
		     boolmatrix const& filtered,
                     double const C1, double const C2,
                     param_struct const& params );
    /** optimal settings for {l,u}_avg */
    void optimizeLU_avg( Eigen::VectorXd const& projection_avg, SparseMb const& y, Eigen::VectorXd const& wc,
                         Eigen::VectorXi const& nclasses, 
			 Eigen::VectorXd const& inside_weight, Eigen::VectorXd const& outside_weight,
			 boolmatrix const& filtered,
                         double const C1, double const C2,
                         param_struct const& params );
    //@}
    
    /** \name Inform about things that have changes (track valid form of items) */
    //@{
    /** update step typically modifies sortlu boundaries (invalidating {l,u}). */
    void chg_sortlu();            // flag a possible change to sortlu, ok_lu --> false
    //@}

    // reset the accumulation
    void reset_acc();
    
    /** \name ensure something is up-to-date before some function call */
    //@{
    void mkok_lu();             ///< if nec. apply changes in sortlu* to l and u
    void mkok_lu_avg();
    void mkok_sortlu();
    void mkok_sortlu_avg();
    Eigen::VectorXd const& l();
    Eigen::VectorXd const& u();
    Eigen::VectorXd const& sortlu();
    Eigen::VectorXd const& l_avg();
    Eigen::VectorXd const& u_avg();
    Eigen::VectorXd const& sortlu_avg();
    std::vector<int> const& perm() const;  ///< forward permutation (ascending 'means') (old 'sorted_class')
    std::vector<int> const& rev() const;   ///< the reverse permutation (old 'class_order')
    
    //@}
    
 private:
    /** Using \c this->perm, generate \c sorted {l,u} pair-vector from \c ll[],uu[] bounds. */
    void toSorted( Eigen::VectorXd & sorted, Eigen::VectorXd const& ll, Eigen::VectorXd const& uu ) const;
    void toLu( Eigen::VectorXd & ll, Eigen::VectorXd & uu, Eigen::VectorXd const& sorted ) const;
 private:
    bool ok_lu;                 ///< after init, one or two of ok_lu and ok_sortlu are always true
    bool ok_sortlu;
    bool ok_lu_avg;             ///< l_avg and u_avg are ok. Set by optimizeLU_avg or by mkok_lu_avg()
    bool ok_sortlu_avg;         ///< sortlu_avg is ok. Set by mkok_sortlu_avg()

    Eigen::VectorXd m_l;                 ///< lower bounds in original class order
    Eigen::VectorXd m_u;                 ///< upper bounds in original class order
    Eigen::VectorXd m_sortlu;            ///< concatenated (l,u) pairs in \c Perm order

    // accumulate sortlu for using with average gradient.  
    Eigen::VectorXd m_l_acc;             ///< used as a convenient temporay for sortlu_acc,
    Eigen::VectorXd m_u_acc;             ///< when reordering classes
    Eigen::VectorXd m_sortlu_acc;
    uint64_t nAccSortlu;    ///< count of accumulations into sortlu_acc from sortlu

    Eigen::VectorXd m_l_avg;             ///< lower bounds in original class order
    Eigen::VectorXd m_u_avg;             ///< sometimes shortly related to \c sortlu_avg
    Eigen::VectorXd m_sortlu_avg;        // average value of sortlu over the accumulation period. 

};


//forward declaration is enough
struct MutexType;

/** solver update step may parallelize by \em chunking the computation. */
struct MCupdateChunking{
    MCupdateChunking( size_t const nTrain, size_t const nClass, size_t const nThreads, param_struct const& p );
    ~MCupdateChunking();
    uint32_t const batch_size;
    /// \name gradient computation of update can be parallized over 'chunks'
    //@{
    int const sc_chunks;
    int const sc_chunk_size;
    int const sc_remaining;
    //@}

    /// \name minibatch update can also have other parallelism
    //@{
    int const idx_chunks;
    int const idx_chunk_size;
    int const idx_remaining;
    MutexType* const idx_locks;
    MutexType* const sc_locks;
    //@}
};
#endif // MCSOLVER_H
