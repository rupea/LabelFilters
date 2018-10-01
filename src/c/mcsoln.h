/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCSOLN_H
#define MCSOLN_H

#include "typedefs.h"
#include "parameter.h"
#include  <array>

/** Corresponds to data stored in an MCFilter "solution" file.
 * - Contains:
 *   - dimensionality data
 *   - {w,l,u} data outputs of time-average solution
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

    /// \name data.
    //@{
private:
    mutable std::array<char,4> magicData;                  ///< MCsc
public:
    DenseColM weights;                 ///< [ d x nProj ] time-avg'd projection matrix
    DenseColM lower_bounds;            ///< [ nClass x nProj ]
    DenseColM upper_bounds;            ///< [ nClass x nProj ]
private:
    mutable std::array<char,4> magicEof1;                  ///< MCs{c|z}
    //@}
};


/** For debug tests: write to sstream, read from sstream, throw if error detected */
void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt);

#endif //MCSOLN_H
