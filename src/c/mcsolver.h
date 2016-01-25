#ifndef MCSOLVER_H
#define MCSOLVER_H
/** \file
 * helper classes to make MCsolver::solve more readable
 */
#include "find_w.h"
#include "boolmatrix.h"

/** MCPRM 0 = no MCpermState.
 * - 1 = define it, use it in easy ways
 * - 2 = use it (eliminate most old vars)
 *   - and then to add 'mk_ok' calls before function calls modifying luPerm variables
 * - 3 = delete original "optimized" conditional xfers from l,u <--> sortlu (and for _avg)
 */
#define MCPRM 3
#if MCPRM < 3
#error "MCPRM < 3 code has been REMOVED"
#endif


class Perm;             ///< an internal detail class -- no user access
class MCpermState;      ///< track which versions of {l,u}{,_avg} and sortedLU{,_avg} are official
class MCiterBools;      ///< t4_XXX bools for "Should we {params.XXX} at iteration MCsoln::t?"
class MCupdateState;    ///< utility constants/variables passed to update routine

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
    void rank( VectorXd const& sortkey );
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
    void rank( VectorXd& sortkey ); // unperm, rerank, set flags

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
    void init( /* inputs: */ VectorXd const& projection, SparseMb const& y, VectorXi const& nc );

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
    void optimizeLU( VectorXd const& projection, SparseMb const& y, VectorXd const& wc,
                     VectorXi const& nclasses, boolmatrix const& filtered,
                     double const C1, double const C2,
                     param_struct const& params, bool print=false );
    /** optimal settings for {l,u}_avg */
    void optimizeLU_avg( VectorXd const& projection_avg, SparseMb const& y, VectorXd const& wc,
                         VectorXi const& nclasses, boolmatrix const& filtered,
                         double const C1, double const C2,
                         param_struct const& params, bool print=false );
    //@}


    /** \name Inform about things that have changes (track valid form of items) */
    //@{
    /** update step typically modifies sortlu boundaries (invalidating {l,u}). */
    void chg_sortlu();            // flag a possible change to sortlu, ok_lu --> false
    /** update step occasionally modify sortlu_avg boundaries (invalidating {l,u}_avg). */
    void chg_sortlu_avg();

    /** optimizeLU, on the other hand, changes {l,u}* (invalidating sortlu*). */
    void chg_lu();
    void chg_lu_avg();
    //@}

    /** \name ensure something is up-to-date before some function call */
    //@{
    void mkok_lu();             ///< if nec. apply changes in sortlu* to l and u
    void mkok_lu_avg();
    VectorXd& mkok_sortlu();
    VectorXd& mkok_sortlu_avg();
    //@}
    
    /** at report time, we \em might want a temporary sortlu list for the hinge loss calc.
     * This calculates and returns client-specified sortlu_avg vector.
     * Note this is const (after this, ok_sortlu_avg has NOT changed) */
    void getSortlu_avg( VectorXd& sortlu_test ) const;
private:
    void toLu( VectorXd & ll, VectorXd & uu, VectorXd const& sorted );
    void toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu );
    void toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ) const;
private:
    //MCsoln & const mcs;
    bool ok_lu;                 ///< after init, one or two of ok_lu and ok_sortlu are always true
    bool ok_lu_avg;             ///< after t4_doing_avg_epoch, at least one of ok_lu_avg and ok_sortlu_avg should be true
    bool ok_sortlu;
    bool ok_sortlu_avg;

    VectorXd l;                 ///< lower bounds in original class order
    VectorXd u;                 ///< upper bounds in original class order
    VectorXd sortlu;            ///< concatenated (l,u) pairs in \c Perm order

    /** tricky dataflow here.
     * - During solve iteration, \c sortlu_avg \b accumulates values from sortlu.
     * - Data flow:
     *   - sortlu --> sortlu_avg   (accumulate during gradient update)
     *     - sortlu_avg --> {l,u}_avg only if req'd for REORDER_AVG_PROJ_MEANS
     *       - if reorder, then go back: {l,u}_avg --> sortlu_avg (if req'd)
     *   - Then near \em end, I might expect
     *     - sortlu_avg no longer relevant
     *     - optimizeLU (or maybe copy {l,u}) --> \em final {l,u}_avg
     */
    VectorXd sortlu_avg;
    uint64_t nAccSortlu_avg;    ///< count of accumulations into sortlu_avg from sortlu

    VectorXd l_avg;             ///< used as a convenient temporay,
    VectorXd u_avg;             ///< sometimes shortly related to \c sortlu_avg
};

/** iteration state that does not need saving -- important stuff is in MCsoln */
struct MCiterBools
{
    /** constructor includes "print some progress" block to cout */
    MCiterBools( uint64_t const t, param_struct const& params );
    bool const reorder;              ///< true if param != 0 && t%param==0
    bool const report;               ///< true if param != 0 && t%param==0
    bool const report_avg;           ///< true if param != 0 && t%param==0
    bool const optimizeLU;           ///< true if param != 0 && t%param==0
    bool const finite_diff_test;     ///< true if param != 0 && t%param==0
    bool const doing_avg_epoch;      ///< avg_epoch && t >= avg_epoch
};

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
