#ifndef MCSOLVER_H
#define MCSOLVER_H
/** \file
 * helper classes to make MCsolver::solve more readable
 */
#include "mcsoln.h"
#include "boolmatrix.h"

/** MCPRM 0 = no MCpermState.
 * - 1 = define it, use it in easy ways
 * - 2 = use it (eliminate most old vars)
 *   - and then to add 'mk_ok' calls before function calls modifying luPerm variables
 * - 3 = delete original "optimized" conditional xfers from l,u <--> sortlu (and for _avg)
 * - etc etc
 */
#define MCPRM 5
#if MCPRM < 5
#error "MCPRM < 5 code has been REMOVED"
#endif


class Perm;             ///< an internal detail class -- no user access
class MCpermState;      ///< track which versions of {l,u}{,_avg} and sortedLU{,_avg} are official
class MCiterBools;      ///< t4_XXX bools for "Should we {params.XXX} at iteration MCsoln::t?"
class MCupdateState;    ///< utility constants/variables passed to update routine

/** Provide a simpler API for solving.
 * - The long \c solve template is now in a 'detail' header,
 * - The library explicitly instiates for input data \c DenseM and SparseM.
 * - So no long headers are required to compile if you link with the library.
 *
 * - \b NOTE: eventually, one might move some of the other utility routines here ?
 */
class MCsolver : protected MCsoln
      //, private MCpermState      // during iteration, sometimes we work with sortedLU_*, other times need l,u
      //, private MCiterBools      // utility bools, now easy to ref in details of solve(..)
{
public:

    /**TODO: fix \b \c solve_optimization so proper resume can be done. */

  //    MCsolver( char const* const solnfile = nullptr );
    MCsolver();
    ~MCsolver();

    //    param_struct const& getParms() const {return this->parms;}
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

#if 0
    /** TBD - For each projection add a \b median value for each class label.
     * This augments {l,u} bounds information, and can be used to provide a
     * \e crude non-binary score for each class.
     *
     * This is a quick'n'easy \e built-in alternative to using a more powerful
     * decision mechanism, like Alex's original one-vs-all SVM to continue the
     * label-filtering process.
     *
     * <B>Begin with just setting the medians </B>
     * This could be \em extended to track a set of quantiles.
     * Quantiles could be compactly stored as \c true_lower,
     * \c true_upper bounds, followed by byte-values[16?]
     * for the approx positions of intermediate quantiles.
     */
    template< typename EIGENTYPE >
        setQuantiles( EIGENTYPE const& x, SparseMb const& y );
#endif

    //    enum Trim { TRIM_LAST, TRIM_AVG };
    /** Free memory by moving selected {w,l,u} data into {w,l,u}_avg.
     * - After a \c solve, or a \c read we may have:
     *   - {w,l,u} of last iteration (and objective_val)
     *   - and {w,l,u}_avg of the time-averaged solution (and objective_val_avg)
     * - But only one set of these is required to \b use the MCsoln
     * - So after \c solve (or maybe \c read),
     *   - you might \c write the LONG/SHORT MCsoln to disk
     *   - and then call trim(...) to free some memory
     * \post MCsoln is a model of SHORT data -- only {w,l,u}_avg might contain data
     *
     * \note While there may be some issue of whether to use w_avg or w, it seems
     * that the correct function for l and u should be to calculate the \b exact
     * lower and upper boundaries for whatever projection axes we choose.
     * This then requires a post-processing pass over the data, during which
     * other easy trivial operations can be done --- like producing an auxiliary
     * class median vector that can be quite useful as a built-in poor-man's
     * nearest-neighbour predictor (for extremely small extra storage).
     * \detail
     * - rename 'postSolve', if it does more than just Trim?
     */
    //    void trim( enum Trim const kp = TRIM_AVG );

private:
    /** twice, we need to chop unused projections from the solution */
    Eigen::VectorXd objective_val;
    void setNProj(uint32_t const nProj, bool, bool);
    int getNthreads( param_struct const& params ) const;
};


/** Maintain a class-permutation and its inverse permutation */
class Perm {
private:
    friend class MCsolver;
    friend class MCpermState; ///< allow access only via MCpermState
    friend class MCupdate;      // perhaps temporarily
    Perm(size_t nClass) : perm(), rev()       ///< initialize to identity permutation
    {
        perm.reserve(nClass);
        rev.reserve(nClass);
        for(int i=0U, n=static_cast<int>(nClass); i<n; ++i){
            perm.push_back(i);
            rev.push_back(i);
        }
    }
    // \todo move back to int, when refactoring done
    std::vector<int> perm;  ///< forward permutation (ascending 'means') (old 'sorted_class')
    std::vector<int> rev;   ///< the reverse permutation (old 'class_order')
    /** produce new perm+rev according to ascending \c sortKey. */
    void rank( Eigen::VectorXd const& sortkey );
};

/** Some data is useful to maintain for post-processing operations
 * after MCsolver::solve has been called.
 */
class MCpermState : private Perm
{
public:
    friend class MCsolver;
    friend class MCupdate;      // perhaps temporarily
    MCpermState( size_t nClass );       ///< allocate 6 vectors of size nClass, nothing is 'ok'
    /** produce new perm+rev according to ascending \c sortKey. */
    void rank( Eigen::VectorXd const& sortkey ); // unperm, rerank, set flags

    /// \name Various ways to initialize \c l and \c u
    //@{
    /** nothing is 'ok', return to freshly-constructed state. */
    void reset();

    /** init lower and upper bounds (lu) per class based on min-max projection values.
     * \p projection    dot-products of training examples with a particular projection axis
     * \p y             class labels for same training examples
     * \p nc            number training examples assigned to each class
     * - A quick substitute for the correct solution of \c optimizeLU.
     * \post \c ok_lu==true and \c ok_sortlu==false
     *
     * - Original: BAD API --- it is only faster once per projection loop and never usable elsewhere
     *   - While doing so, return a \c means vector according to \c reorder
     *   - \c reorder == \c REORDER_AVG_PROJ_MEANS is treated as \c REORDER_PROJ_MEANS
     */
    void init( /* inputs: */ Eigen::VectorXd const& projection, SparseMb const& y, Eigen::VectorXi const& nc );

    /** If restarting from soln file, can also explicitly set initial state */
    template< typename EIGENTYPE1, typename EIGENTYPE2 >
    void set_lu( EIGENTYPE1 const& ll, EIGENTYPE2 const& uu  ){
        assert( ll.size() == uu.size() );
        l = ll;
        u = uu;
        ok_lu = true;
        ok_sortlu = false;
    }

    /** optimal settings for {l,u} */
    void optimizeLU( Eigen::VectorXd const& projection, SparseMb const& y, Eigen::VectorXd const& wc,
                     Eigen::VectorXi const& nclasses, boolmatrix const& filtered,
                     double const C1, double const C2,
                     param_struct const& params, bool print=false );
    /** optimal settings for {l,u}_avg */
    void optimizeLU_avg( Eigen::VectorXd const& projection_avg, SparseMb const& y, Eigen::VectorXd const& wc,
                         Eigen::VectorXi const& nclasses, boolmatrix const& filtered,
                         double const C1, double const C2,
                         param_struct const& params, bool print=false );
    //@}

    // Accumulate the current sortlu into sortlu_acc and increaset nAccSortlu
    //    void accumulate_sortlu();
    
    /** \name Inform about things that have changes (track valid form of items) */
    //@{
    /** update step typically modifies sortlu boundaries (invalidating {l,u}). */
    void chg_sortlu();            // flag a possible change to sortlu, ok_lu --> false

    /** optimizeLU, on the other hand, changes {l,u}* (invalidating sortlu*). */
    //    void chg_lu();
    //@}

    /** \name ensure something is up-to-date before some function call */
    //@{
    void mkok_lu();             ///< if nec. apply changes in sortlu* to l and u
    void mkok_lu_avg();
    Eigen::VectorXd& mkok_sortlu();
    Eigen::VectorXd& mkok_sortlu_avg();
    //@}

    /** Using \c this->perm, generate \c sorted {l,u} pair-vector from \c ll[],uu[] bounds. */
    void toSorted( Eigen::VectorXd & sorted, Eigen::VectorXd const& ll, Eigen::VectorXd const& uu ) const;
private:
    void toLu( Eigen::VectorXd & ll, Eigen::VectorXd & uu, Eigen::VectorXd const& sorted );
    //void toSorted( Eigen::VectorXd & sorted, Eigen::VectorXd const& ll, Eigen::VectorXd const& uu );
private:
    //MCsoln & const mcs;
    bool ok_lu;                 ///< after init, one or two of ok_lu and ok_sortlu are always true
    bool ok_lu_avg;             ///< after t4_doing_avg_epoch, at least one of ok_lu_avg and ok_sortlu_avg should be true
    bool ok_sortlu;
    bool ok_sortlu_avg;

    Eigen::VectorXd l;                 ///< lower bounds in original class order
    Eigen::VectorXd u;                 ///< upper bounds in original class order
    Eigen::VectorXd sortlu;            ///< concatenated (l,u) pairs in \c Perm order

    // accumulate sortlu for using with average gradient.  
    Eigen::VectorXd sortlu_acc;
    uint64_t nAccSortlu;    ///< count of accumulations into sortlu_acc from sortlu

    Eigen::VectorXd l_avg;             ///< used as a convenient temporay,
    Eigen::VectorXd u_avg;             ///< sometimes shortly related to \c sortlu_avg
    Eigen::VectorXd sortlu_avg;        // average value of sortlu over the accumulation period. 

};

/** iteration state that does not need saving -- important stuff is in MCsoln */
struct MCiterBools
{
  /** constructor includes "print some progress" block to cout */
  MCiterBools( uint64_t const t, param_struct const& params );
  bool const reorder;              ///< true if param != 0 && t%param==0
  bool const report;               ///< true if param != 0 && t%param==0
  bool const optimizeLU;           ///< true if param != 0 && t%param==0
  bool const doing_avg_epoch;      ///< avg_epoch && t >= avg_epoch
  bool const progress;             ///< params.verbose >= 1 && !params.report_epoch && t % 1000 == 0
#if GRADIENT_TEST
  bool const finite_diff_test;     ///< true if param != 0 && t%param==0
#endif
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
