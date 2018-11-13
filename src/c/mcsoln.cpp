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
#define RETHROW( ERRMSG ) std::cerr<<e.what()<<std::endl;; throw std::runtime_error(ERRMSG)

void MCsoln::write_ascii( std::ostream& os) const
{
  try{
    eigen_io_txt(os,weights);
    eigen_io_txt(os,lower_bounds);
    eigen_io_txt(os,upper_bounds);
  } catch(exception const& e){ RETHROW("MCsoln data write error"); }
}

void MCsoln::read_ascii( std::istream& is ){
  try{
    eigen_io_txt(is,weights);
    eigen_io_txt(is,lower_bounds);
    eigen_io_txt(is,upper_bounds);    
    d = weights.rows();
    nProj = weights.cols();
    nClass = lower_bounds.rows();
    if(lower_bounds.cols() != nProj || upper_bounds.cols() != nProj || upper_bounds.rows() != nClass)
      {
	throw runtime_error("MCsoln dimensions missmatch");
      }
  } catch(exception const& e){ RETHROW("MCsoln data read error");}
}

void MCsoln::write_binary( std::ostream& os) const
{
  try{
    eigen_io_bin(os,weights);
    eigen_io_bin(os,lower_bounds);
    eigen_io_bin(os,upper_bounds);
    io_bin(os,magicEof);
  } catch(exception const& e){ RETHROW("MCsoln data write error"); }
}
void MCsoln::read_binary( std::istream& is ){
  try{
    eigen_io_bin(is,weights);
    eigen_io_bin(is,lower_bounds);
    eigen_io_bin(is,upper_bounds);
    io_bin(is,magicEof1);
    if(!MAGIC_EQU(magicEof1,magicEof)) throw runtime_error("MCsoln bad data terminator");
    d = weights.rows();
    nProj = weights.cols();
    nClass = lower_bounds.rows();
    if(lower_bounds.cols() != nProj || upper_bounds.cols() != nProj || upper_bounds.rows() != nClass)
      {
	throw runtime_error("MCsoln dimensions missmatch");
      }
  } catch(exception const& e){ RETHROW("MCsoln data read error"); }
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
