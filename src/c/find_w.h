#ifndef __FIND_W_H
#define __FIND_W_H

#include "typedefs.h"
#include "parameter.h"
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
 *  \p objective_val    out:  VectorXd(k)
 *  \p w_avg            inout: d x p matrix, init setRandom
 *  \p l_avg            inout: k x p matrix, init setZero
 *  \p u_avg            inout: k x p matrix, init setZero
 *  \p object_val_avg   out: VectorXd(k)
 *  \p x                in: X x d, X d-dim training examples
 *  \p y                in: X x 1, X labels of each training example
 *  \p params           many parameters, eg from set_default_params()
 *
 *  - The library has instantiations for EigenType:
 *    - DenseM and
 *    - SparseM
 */

template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds,
                        VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg,
                        VectorXd& objective_val_avg,
                        const EigenType& x, const SparseMb& y,
                        const param_struct& params);

#if 1 // proposed for lua api.
/** Corresponds to data stored in an MCFilter "solution" file.
 * - Contains:
 *   - dimensionality data & "short" or "long" format spec
 *   - \c param_struct of last call to MCsolveMCsolver::solve()
 *   - some <em>final iteration</em> values
 *   - {w,l,u,obj} data outputs of time-average solution
 *   - [opt. if len=="long"] {w,l,u,obj} of final iteration
 * - during MCsolver::solve, we need the "long" format,
 *   but the solution can still be saved to disk in "short" 
 */
class MCsoln {
public:
    enum Len : char { /*default*/SHORT, LONG };
    enum Fmt : char { /*default*/BINARY, TEXT };

    /** Construct to begin a solution from scratch.
     * - Unknown hdr dims filled in during solve (know training data)
     * - Avg weights set to random values
     */
    MCsoln();

    /** Construct from solution file. */
    MCsoln( char const* solnfile );

    /// \name i/o, throw on error
    //@{
public:
    void read( std::istream& is );
    void write( std::ostream& os, enum Fmt fmt=BINARY, enum Len len=SHORT ) const;

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
private:
    mutable std::array<char,4> magicEof1;                  ///< MCs{c|z}
public:
    //VectorXd& objective_val_avg; // hmmm. this is optional, I guess
    //@}
    /// \name Len==LONG data.
    /// - Optional -- can be written/read as empty vectors
    /// - resized as neces
    //@{
    VectorXd objective_val_avg;         ///< [ nClass ] objective values
private:
    mutable std::array<char,4> magicEof2;                  ///< MCs{c|z} no enum Len for ending here [yet]
public:

    DenseM weights;                     ///< final iteration data
    DenseM lower_bounds;                ///< final iteration data
    DenseM upper_bounds;                ///< final iteration data
    VectorXd objective_val;             ///< final iteration data
private:
    mutable std::array<char,4> magicEof3;                  ///< MCsz
public:
    //@}

};

/** for debug tests: write to sstream, read from sstream, throw if error detected */
void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt, enum MCsoln::Len len);

class MCsolver : public MCsoln {
public:

    /** Initialize with given input data.
     * If resume data is OK (readable, compatible with x, y) invoke
     * solve_optimization with appropriate initialization.
     *
     * TODO: fix \b \c solve_optimization so proper resume can be done.
     *       Initially just copy solve_optimization code until we can
     *       deprecate the original (which is needed for the octave api).
     */
    MCsolver( char const* solnfile = nullptr );
    ~MCsolver();

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
     */
    template< typename EIGENTYPE >
        int solve( EIGENTYPE const& x, SparseMb const& y, param_struct const* = nullptr );
};
#endif // proposed
#endif // __FIND_W_H
