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
    void write( std::ostream& os, enum Fmt fmt=BINARY ) const;
    void pretty( std::ostream& os, int verbose = 0 ) const; ///< short'n'sweet dump of main content

private: // after the 4-byte magic header we specialize I/O routines
    void read_ascii( std::istream& is );
    void read_binary( std::istream& is );
    void write_ascii( std::ostream& is) const;
    void write_binary( std::ostream& is) const;
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
    //@}

    /// \name parameters used for last/current call to MCsolver::solve(...)
    /// not used for now because it invalidates saved solutions if new parameters are added
    //@{
    //    param_struct parms;
    //@}

    /// \name data.
    //@{
private:
    mutable std::array<char,4> magicData;                  ///< MCsc
public:
    DenseM weights;                 ///< [ d x nProj ] time-avg'd projection matrix
    DenseM lower_bounds;            ///< [ nClass x nProj ]
    DenseM upper_bounds;            ///< [ nClass x nProj ]
private:
    mutable std::array<char,4> magicEof1;                  ///< MCs{c|z}
    //@}
};


/** For debug tests: write to sstream, read from sstream, throw if error detected */
void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt);

#endif //MCSOLN_H
