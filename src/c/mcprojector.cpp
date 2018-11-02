/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mcprojector.h"
#include "mcxydata.h"
#include "mcfilter.h"
#include "printing.hh"
#include "utils.h"              // OUTWIDE

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory> //shared_ptr

using namespace std;

MCprojector::MCprojector ( DP data, FP lf /* = FP()*/, int const vb /*=0*/)
  : m_xy( data )
  , m_filter(lf)
  , m_feasible(nullptr)
  , feasible_ok(false)
  , m_nProj(lf->nFilters())
  , m_nfeasible(0)
  , verbose(vb)
{}

MCprojector::~MCprojector()
{
  if (m_feasible) delete m_feasible;
}

void MCprojector::runFilter()
{
  if(verbose>=1) cout<<" MCprojProgram::runFilter() "<<(m_xy->denseOk?"dense":m_xy->sparseOk?"sparse":"HUH?")<<endl;
  
  if(!m_xy)
    {
      throw runtime_error("runFilter() called without data");
    }
  
  if (!m_filter || m_nProj == 0)
    {
      // no filter or 0 projections requested. Set feasible to null. 
      if (m_feasible)
	{
	  delete m_feasible;
	  m_feasible = nullptr;
	  m_nfeasible = 0; //not calculate yet. Will calculate later if needed.
	  feasible_ok=true;
	}
      return;
    }

  if (!m_feasible) m_feasible = new ActiveSet();
  if( m_xy->denseOk ){
    m_filter->filter(*m_feasible, m_xy->xDense, m_nProj );
  }else if( m_xy->sparseOk ){
    m_filter->filter(*m_feasible, m_xy->xSparse, m_nProj );
  }else{
    if (m_feasible) delete m_feasible;
    feasible_ok = false;
    m_nfeasible = 0;
    throw std::runtime_error("neither sparse nor dense data x was available");
  }
  feasible_ok=true;
  m_nfeasible = 0;//not calculated yet. Will calculate later if needed. 
}

void MCprojector::saveFeasible( std::string const& fname /*=""*/, bool binary /*= false*/)
{
  if(verbose>=1) cout<<" MCprojProgram::saveFeasible()"
		     <<"\tSaving feasible "<<(binary?"Binary":"Text")<<"-->"
		     <<(fname.size()? fname: string("cout"))
		     <<endl;
  
  if (!feasible_ok)
    {
      throw runtime_error("Trying to save the feasible set, but it has not been computed yet. Try running filter()");
    }
  
  ofstream ofs;
  if (fname.size()==0){
    if (binary){
      cerr<<"Warning: Binary flag ignored for cout"<<endl;
      binary = false;
    }
  }else{
    ofs.open(fname);
  }
  ostream& out = fname.size()?ofs:cout;
  if(!out.good() ) throw std::runtime_error("saveFeasible: trouble opening file");
  
  if(binary){
    detail::io_bin(out, m_feasible);
    if(verbose>=2) cout<<"Note:\tIn C++, use printing.hh\n\t\tio_bin(ifstream(\""<<fname
		       <<"\"),vector<Roaring>&)\n\tcan read the projections binary file";
  }else{ // outText
    detail::io_txt(out, m_feasible);
    if( ! out.good() ) throw std::runtime_error("trouble writing outFile");
  }
  if (fname.size()) ofs.close();
  
}


void MCprojector::nFeasible(size_t& nfeasible, double& prc_feasible)
{
  if (!feasible_ok)
    {
      runFilter();
    }

  size_t total = m_xy->y.rows()*m_xy->y.cols();
  if ( m_nfeasible == 0 ) // has not been calculated
    {
      if (m_feasible)
	{
	  for (ActiveSet::iterator it = m_feasible->begin(); it < m_feasible->end();it++)
	    {
	      m_nfeasible += (*it).cardinality();
	    }
	}
      else
	{
	  m_nfeasible = total;
	}      
    }
  nfeasible = m_nfeasible;
  prc_feasible = nfeasible*1.0/total;
}
			   
