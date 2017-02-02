#ifndef __FIND_W_H
#define __FIND_W_H

#include "typedefs.h"
#include "parameter.h"
#include "mutexlock.h"  // deprecate this XXX (use std::mutex?)
//#include "mcsolver.h"   // maybe just the fwd declarations?
#include <iosfwd>
#include  <array>

/**
 * Solve the optimization using the gradient descent on hinge loss.
 *
 * - Suppose \c d-dimensional features, \c k classes, and \c p features
 * - The training data \c x is a <examples> x \c d matrix [k cols]
 * - The training labels \c y is a <examples> x \c k matrix [k cols],
 *   - typically a vector converted to a SparseMb (sparse bool matrix)
 *  \p weights          inout: d x p matrix, init setRandom
 *  \p lower_bounds     inout: k x p matrix, init setZero
 *  \p upper_bounds     inout: k x p matrix, init setZero
 *  \p objective_val    out:  VectorXd(k...) (a growing history)
 *  \p w_avg            inout: d x p matrix, init setRandom
 *  \p l_avg            inout: k x p matrix, init setZero
 *  \p u_avg            inout: k x p matrix, init setZero
 *  \p object_val_avg   out: VectorXd(k) (a growing history)
 *  \p x                in: X x d, X d-dim training examples
 *  \p y                in: X x 1, X labels of each training example
 *  \p params           many parameters, eg from set_default_params()
 *
 *  - The library has instantiations for EigenType:
 *    - DenseM and
 *    - SparseM
 */

template<typename EIGENTYPE>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, Eigen::VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, Eigen::VectorXd& objective_val_avg,
                        EIGENTYPE const& x,
                        SparseMb const& y,
                        param_struct const& params);

#if 1 // proposed for lua api.
/** Corresponds to data stored in an MCFilter "solution" file.
 * - Contains:
 *   - dimensionality data
 *   - \c param_struct of last call to MCsolveMCsolver::solve()
 *   - some <em>final iteration</em> values
 *   - {w,l,u,obj} data outputs of time-average solution
 *   - [opt. if len=="long"] {w,l,u,obj} of final iteration
 * - during MCsolver::solve, we expand data to LONG format,
 *   - but the solution can still be saved to disk in SHORT
 *   - and after \c solve, we can \c keep() just a \c SHORT data
 * - other operations (TBD) only need the \c SHORT data
 */
class MCsoln {
public:
    enum Len : char {
        /*default*/SHORT,       ///< retain just {w,l,u}_avg matrices of the solution
        LONG                    ///< But during \c solve, we use {w,l,u} and objective_vals*.
    };
    enum Fmt : char { /*default*/BINARY, TEXT };

    /** Construct to begin a solution from scratch.
     * - Unknown hdr dims filled in during solve (know training data)
     * - Avg weights set to random values
     */
    MCsoln();

    // /** Construct from solution file. */
    // MCsoln( char const* solnfile );

    /// \name i/o, throw on error
    //@{
public:
    void read( std::istream& is );
    void write( std::ostream& os, enum Fmt fmt=BINARY, enum Len len=SHORT ) const;
    void pretty( std::ostream& os ) const; ///< short'n'sweet dump of main content

private: // after the 4-byte magic header we specialize I/O routines
    void read_ascii( std::istream& is );
    void read_binary( std::istream& is );
    void write_ascii( std::ostream& is, enum Len len=SHORT ) const;
    void write_binary( std::ostream& is, enum Len len=SHORT ) const;
    static std::array<char,4> magicTxt; ///< MCst text?
    static std::array<char,4> magicBin; ///< MCsb binary?
    static std::array<char,4> magicCnt; ///< MCsc continue?
    static std::array<char,4> magicEof; ///< MCsz eof?
public:
    //@}
    // -------- data layout --------
    /// \name header, esp dimensionality constants for binary save/restart data.
    /// These always match the matrix/vector data
    //@{
private:
    mutable std::array<char,4> magicHdr;        ///< required -- MCst / MCsb text or binary ?
public:
    uint32_t d;                         ///< d is x.cols example dimensionality
    uint32_t nProj;                     ///< w is d x nProj
    uint32_t nClass;                    ///< l and u are nClass x nProj, objective[nClass]
    std::string fname;                  ///< name of solution file (or empty)
    //@}

    /// \name parameters used for last/current call to MCsolver::solve(...)
    //@{
    param_struct parms;
    //@}

    /// \name restart constants pertinent to <em>final iteration t</em>.
    /// - These can be used to resume/extend a previous solution.
    /// - Is there a param_struct::resume setting to ignore or use these?
    //@{
    uint64_t t;                         ///< iteration number
    double C1;                          ///< regularization constant
    double C2;                          ///< regularization constant
    double lambda;                      ///< factor governing decay of C1 or C2
    double eta_t;                       ///< learning rate
    // add more, moving the solve_optimization local vars up here ...
    //@}

    /// \name Len==SHORT data.
    //@{
private:
    mutable std::array<char,4> magicData;                  ///< MCsc
public:
    DenseM weights_avg;                 ///< [ d x nProj ] time-avg'd projection matrix
    DenseM lower_bounds_avg;            ///< [ nClass x nProj ]
    DenseM upper_bounds_avg;            ///< [ nClass x nProj ]
    DenseM medians;                     ///< [ nClass x nProj ] [opt] set during 'solve' post-processing
private:
    mutable std::array<char,4> magicEof1;                  ///< MCs{c|z}
public:
    //Eigen::VectorXd& objective_val_avg; // hmmm. this is optional, I guess
    //@}
    /// \name Len==LONG data.
    /// - Optional -- can be written/read as empty vectors
    /// - resized as neces
    //@{
    Eigen::VectorXd objective_val_avg;         ///< [ nClass ] objective values
private:
    mutable std::array<char,4> magicEof2;                  ///< MCs{c|z} no enum Len for ending here [yet]
public:

    DenseM weights;                     ///< final iteration data
    DenseM lower_bounds;                ///< final iteration data
    DenseM upper_bounds;                ///< final iteration data
    Eigen::VectorXd objective_val;             ///< final iteration data
private:
    mutable std::array<char,4> magicEof3;                  ///< MCsz
public:
    //@}

};


/** For debug tests: write to sstream, read from sstream, throw if error detected */
void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt, enum MCsoln::Len len);

/** lazy stats for x,y data. WIP, so opaque for now. */
struct MCLazyData;

/** NEW: introduce a data wrapper.
 *
 * - This allows base x [,y] data to be passed/shared easily between solver
 *   and projector objects, \c MCprojProg and \c MCsolveProgram.
 * - Another benefit is removing duplicate code for reading data.
 * - Eventually, may help move template code into the library? */
class MCxyData {
public:
    MCxyData();
    ~MCxyData();
    /// \name row-wise test data matrix
    //@{
    // perhaps denseOk and sparseOk can be replaced by xDense.size() != 0 (etc.) ?
    DenseM xDense;
    bool denseOk;
    SparseM xSparse;
    bool sparseOk;
    //@}
    SparseMb y;                 ///< optional for projection operation.
    /// \name optional, private stats
    //@{
private:
    double qscal; ///< if >0, the multiplier used for \c quadx dimensions
    double xscal; ///< if >0, the global x multipler used for \c xscale
    struct MCLazyData * lazx;   ///< lazy x statistics
    struct MCLazyData * lazy;   ///< lazy y statistics
public:
    void xchanged();                    ///< scrap any x stats
    void ychanged();                    ///< scrap any y stats
    //@}

    void xread( std::string xFile );    ///< read x (binary, sparse/dense) (txt fmt \b todo)
    void yread( std::string yFile );    ///< read x (sparse binary or text)
    void xwrite( std::string xFile ) const; ///< save x (binary only, for now)
    void ywrite( std::string yFile ) const; ///< save y (binary only, for now)

    std::string shortMsg() const;       ///< format+dimensions

    void xrunit();                      ///< scale rows have unit norm.
    void xcunit();                      ///< scale rows have unit norm.

    void xrnormal();                    ///< remove mean,stdev from x rows (dense only)
    void xcnormal();                    ///< remove mean,stdev from x cols (dense only)

    void xscale(double scal);           ///< multiply all x values by const
    double xmul() const {return xscal;} ///< what's global x multiplier?

    void quadx(double qscal=0.0);       ///< add quadratic dimensions (0.0 autoscales, somehow) \throw if no x data
    double quadmul() const {return qscal;} ///< return the used quadmul (or 0.0 if quadx has not been called)
    
    // I got annoyed with weighting for an error before aborting trying binary
    // reads. So let me (everywhere, sigh) use a magic header, for a quick check.
    static std::array<char,4> magic_xSparse; ///< 0x00,'X','s','8' (or 4 for floats)
    static std::array<char,4> magic_xDense;  ///< 0x00,'X','d','8' (or 4 for floats)
    static std::array<char,4> magic_yBin;    ///< 0x00,'Y','s','b'
    // feel free to add any other [binary] formats.
};

/** Provide a simpler API for solving.
 * - The long \c solve template is now in a 'detail' header,
 * - The library explicitly instiates for input data \c DenseM and SparseM.
 * - So no long headers are required to compile if you link with the library.
 *
 * - \b NOTE: eventually, one might move some of the other utility routines here ?
 */
class MCsolver : protected MCsoln
      // begin with these "shadowing" the original variables, until exact same function
      // is verified.
      //, private MCpermState      // during iteration, sometimes we work with sortedLU_*, other times need l,u
      //, private MCiterBools      // utility bools, now easy to ref in details of solve(..)
{
public:

    /** Initialize with given input data.
     * If resume data is OK (readable, compatible with x, y) invoke
     * solve_optimization with appropriate initialization.
     *
     * TODO: fix \b \c solve_optimization so proper resume can be done.
     *       Initially just copy solve_optimization code until we can
     *       deprecate the original (which is needed for the octave api).
     */
    MCsolver( char const* const solnfile = nullptr );
    ~MCsolver();

    param_struct const& getParms() const {return this->parms;}
    MCsoln       const& getSoln()  const {return *this;}
    MCsoln            & getSoln()        {return *this;}
    void read( std::istream& is )
    { MCsoln::read(is); }
    void write( std::ostream& os, enum Fmt fmt=BINARY, enum Len len=SHORT ) const
    { MCsoln::write(os,fmt,len); }

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
        void solve( EIGENTYPE const& x, SparseMb const& y, param_struct const* const params_arg = nullptr );

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

    enum Trim { TRIM_LAST, TRIM_AVG };
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
    void trim( enum Trim const kp = TRIM_AVG );

private:
    /** twice, we need to chop unused projections from the solution */
    void chopProjections(size_t const nProj);
    int getNthreads( param_struct const& params ) const;
};
#endif // proposed
#endif // __FIND_W_H
