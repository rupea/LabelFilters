/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mcsoln.h"
#include "printing.hh" 
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>


// -------- MCsoln I/O --------
using namespace std;
using namespace detail;     // see printing.h

std::array<char,4> MCsoln::magicBin = {'M', 'C', 's', 'b' };
std::array<char,4> MCsoln::magicCnt = {'M', 'C', 's', 'c' };
std::array<char,4> MCsoln::magicEof = {'M', 'C', 's', 'z' };

#define MAGIC_EQU( A, B ) (A[0]==B[0] && A[1]==B[1] && A[2]==B[2] && A[3]==B[3])

MCsoln::MCsoln()
: magicHdr( magicBin )
    , d( 0U )           // unknown until first call to 'solve', when have training examples 'x'
    , nProj( 0U )
    , nClass( 0U )

    , magicData( magicCnt )

    , weights()
    , lower_bounds()
    , upper_bounds()
    , magicEof1( magicEof )    
{};
void MCsoln::write( std::ostream& os, enum Fmt fmt/*=BINARY*/) const
{
    if(fmt!=BINARY){
        write_ascii(os);
    }else{
        io_bin(os,magicBin);
        write_binary(os);
    }
}

void MCsoln::read( std::istream& is ){
    io_bin(is,magicHdr);
    if(      MAGIC_EQU(magicHdr,magicBin) ) read_binary( is );
    else
      {
	is.seekg(ios::beg);
	read_ascii(is);
      }
}
void MCsoln::pretty( std::ostream& os, int verbose /*=0*/) const {
  if (verbose >= 1) {
    os<<"--------- d="<<d<<" nProj="<<nProj<<" nClass="<<nClass<<endl;
    os<<"--------- weights"<<prettyDims(weights)
        <<" lower_bounds"<<prettyDims(lower_bounds)
        <<" upper_bounds"<<prettyDims(upper_bounds)<<endl;
  }
  if (verbose >= 2){
    os<<"--------- weights:\n"<<weights<<endl;
    os<<"--------- lower_bounds:\n"<<lower_bounds<<endl;
    os<<"--------- upper_bounds:\n"<<upper_bounds<<endl;
  }
}


// private post-magicHdr I/O routines ...
#define CHK_MAT_DIM(MAT,R,C,ERRMSG) do { \
    if( MAT.rows() != (R) || MAT.cols() != (C) ){ \
        std::ostringstream oss; \
        oss<<ERRMSG<<"\n\tmatrix["<<MAT.rows()<<"x"<<MAT.cols()<<"] expected ["<<(R)<<"x"<<(C)<<"]\n"; \
        throw std::runtime_error(oss.str()); \
    }}while(0)
#define CHK_VEC_DIM(VEC,SZ,ERRMSG) do { \
    if( VEC.size() != (SZ) ) { \
        std::ostringstream oss; \
        oss<<ERRMSG<<"\n\tvector["<<VEC.size()<<"] expected ["<<(SZ)<<"]\n"; \
        throw std::runtime_error(oss.str()); \
    }}while(0)
#define RETHROW( ERRMSG ) std::cerr<<e.what(); throw std::runtime_error(ERRMSG)

void MCsoln::write_ascii( std::ostream& os) const
{
    try{
      io_txt(os,d, " ");
      io_txt(os,nProj, " ");
      io_txt(os,nClass, "\n");
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data write error");
    }

#define IO_MAT( OS, MAT, R, C, ERRMSG ) do{eigen_io_txt(OS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
    try{
        IO_MAT(os, weights    , d     , nProj, "write_ascii Bad weights dimensions");
        IO_MAT(os, lower_bounds, nClass, nProj, "write_ascii Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds, nClass, nProj, "write_ascii Bad upper_bounds dimensions");
    } catch(exception const& e){ RETHROW("MCsoln data write error"); }
#undef IO_MAT
}

void MCsoln::read_ascii( std::istream& is ){
    try{
        io_txt(is,d);
        io_txt(is,nProj);
        io_txt(is,nClass);
	is >> ws;
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data read error");
    }
#define IO_MAT( IS, MAT, R, C, ERRMSG ) do{eigen_io_txt(IS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
    try{
        IO_MAT(is, weights     , d     , nProj, "read_ascii Bad weights dimensions");
        IO_MAT(is, lower_bounds, nClass, nProj, "read_ascii Bad lower_bounds dimensions");
        IO_MAT(is, upper_bounds, nClass, nProj, "read_ascii Bad upper_bounds dimensions");
    } catch(exception const& e){ RETHROW("MCsoln data read error"); }
#undef IO_MAT
}

void MCsoln::write_binary( std::ostream& os) const
{
    try{
        io_bin(os,d);
        io_bin(os,nProj);
        io_bin(os,nClass);
        magicData = magicCnt;
        io_bin(os,magicData);
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data write error");
    }

#define IO_MAT( OS, MAT, R, C, ERRMSG ) do{eigen_io_bin(OS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
    try{
        IO_MAT(os, weights     , d     , nProj, "write_binary Bad weights dimensions");
        IO_MAT(os, lower_bounds, nClass, nProj, "write_binary Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds, nClass, nProj, "write_binary Bad upper_bounds dimensions");
        magicEof1 = magicEof;
        io_bin(os,magicEof1);
    } catch(exception const& e){ RETHROW("MCsoln data write error"); }
#undef IO_MAT
}
void MCsoln::read_binary( std::istream& is ){
    try{
        io_bin(is,d);
        io_bin(is,nProj);
        io_bin(is,nClass);
        io_bin(is,magicData);
        if( ! MAGIC_EQU(magicData,magicCnt) )
            throw runtime_error("MCsoln header length has changed");
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data read error");
    }
#define IO_MAT( IS, MAT, R, C, ERRMSG ) do{eigen_io_bin(IS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
    try{
        IO_MAT(is,      weights, d     , nProj, "read_binary Bad weights dimensions");
        IO_MAT(is, lower_bounds, nClass, nProj, "read_binary Bad lower_bounds dimensions");
        IO_MAT(is, upper_bounds, nClass, nProj, "read_binary Bad upper_bounds dimensions");
        io_bin(is,magicEof1);
        if(!MAGIC_EQU(magicEof1,magicEof)) throw runtime_error("MCsoln bad data terminator");
    } catch(exception const& e){ RETHROW("MCsoln data read error"); }
    return;
#undef IO_MAT
}

void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt)
{
    stringstream ss;
    ostream& os = ss;
    mcsoln.write( os, fmt);
    os.flush();
    MCsoln x;
    istream& is = ss;
    x.read( is );
    cout<<" testMCsolnWriteRead OK, XXX correctness tests TBD !!!"<<endl;
    return;
}
