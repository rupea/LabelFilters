#ifndef __FIND_W_H
#define __FIND_W_H

#include "typedefs.h"
#include "parameter.h"
#include <iosfwd>

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
template< class EigenType >
class MCSolver {
public:

    /** Initialize with given input data.
     * If resume data is OK (readable, compatible with x, y) invoke
     * solve_optimization with appropriate initialization.
     *
     * TODO: fix \b \c solve_optimization so proper resume can be done.
     *       Initially just copy solve_optimization code until we can
     *       deprecate the original (which is needed for the octave api).
     */
    MCSolver( EigenType const& x, SparseMb const& y,
              char const* rfile = nullptr);
    ~MCSolver();

private:
    /// \name dimensionality constants for binary save/restart data.
    /// These always match the matrix/vector data
    //@{
    size_t d;                   ///< d is x.cols example dimensionality
    size_t nProj;               ///< w is d x nProj
    size_t nClass;              ///< l and u are nClass x nProj, objective[nClass]
    std::string fmt;            ///< "short"(avg only) or "long" (avg + time-t data)
    //@}
    /// \name restart constants pertinent to <em>final iteration t</em>
    /// These can optionally be used to resume/extend a previous solution.
    /// There should be a param_struct::resume setting to ignore or use these.
    //@{
    unsigned long t;    ///< iteration number
    double C1;          ///< regularization constant
    double C2;          ///< regularization constant
    double lambda;      ///< factor governing decay of C1 or C2
    double eta_t;       ///< learning rate
    // probably more
    //@}

    /** read/write dimensionality and final iteration t settings.
     * This header data is just a few bytes, so even if we never resume/continue
     * the optimization, it's cheap to keep around. */
    void rfile_hdr_ascii( std::istream& is );
    void rfile_hdr_binary( std::istream& is );
    void rfile_hdr_ascii( std::ostream& is );
    void rfile_hdr_binary( std::ostream& is );
    /** read/write binary {w,l,u,obj} data blob[s].
     * - If \c fmt == <em>"short"</em> then read/write just time-averaged data blob
     * - If \c fmt == <em>"long"</em> then read the unaveraged version of the blob too.
     *   - <em>"long"</em> is about TWICE as long as <em>"short"</em> !
     *   - reading <em>"long"</em> ignores missing <em>"long"</em> data.
     *   - i.e. init the time t data equiv to the time-averaged data
     */
    void rfile_data_ascii( std::istream& is, std::string const fmt );
    void rfile_data_binary( std::istream& is, std::string const fmt );
    void rfile_data_ascii( std::ostream& is, std::string const fmt );
    void rfile_data_binary( std::ostream& is, std::string const fmt );

public:
    int solve( char const* rfile = nullptr, char const* fmt="short" );
};
#endif // proposed
#endif // __FIND_W_H
