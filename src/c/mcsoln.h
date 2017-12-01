#ifndef MCSOLN_H
#define MCSOLN_H

#include "typedefs.h"
#include "parameter.h"
#include  <array>

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
    double C2;                          ///< regularization constant
    double C1;                          ///< regularization constant
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

#endif //MCSOLN_H
