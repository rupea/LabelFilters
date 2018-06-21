#include "mcprojector.h"
#include "mcxydata.h"
#include "mcfilter.h"
#include "printing.hh"
//#include "normalize.h"
//#include "mcpredict.hh"
#include "utils.h"              // OUTWIDE

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory> //shared_ptr

using namespace std;

MCprojector::MCprojector ( DP data, FP lf /* = FP()*/, int const nf /*=-1*/, int const vb /*=0*/)
  : m_xy( data )
  , m_filter(lf)
  , m_feasible(nullptr)
  , feasible_ok(false)
  , m_nProj(nf)
  , m_nfeasible(0)
  , verbose(vb)
{}

MCprojector::~MCprojector()
{
  if (m_feasible) delete m_feasible;
}


  /* this needs to be moved somewhere else
  void MCprojProgram::readData(std::string const& xfile)
  {
    // obtain this->xy->MCxyData::{xDense, denseOk, xSparse,sparseOk}
    xy = std::make_shared<::MCxyData>();
    if(verbose>=1) cout<<"Reading from xfile="<<xfile<<endl;
    xy->xread(xfile);    // tries binary dense, sparse, libsvm, XML            
    if( A::xnorm ){
      // SHOULD USE THE MEAN/SDEV CALCULATED FOR THE TRAINIGN SET, NOT THE ONES FOR THE PROJECTION SET!!!
      ::Eigen::VectorXd xmean;
      ::Eigen::VectorXd xstdev;
      if(verbose>=1){
	cout<<" xnorm!"<<endl;
      }
      //for sparse data do not center (should be an option)
      cerr << "Warning: column normalization is done using statistics on the prediction set.\n"
	   << "         Training set statistics should probably be used" << endl
	   << endl; 
      if (xy->sparseOk)
	cerr << "Warning: --xnorm is used with sparse data. Features are not centered to maintain sparsity" << endl;	      
      xy->xstdnormal(xmean, xstdev, true, !xy->sparseOk, false);
      if(verbose>=1){
	cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
	cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
      }
    }
    
    if( A::xunit ){
      xy->xunitnormal();
    }
    
    if( A::xscale != 1.0 ){
      xy-> xscale( A::xscale );
    }
    
    if(verbose>=1){
      if( xy->denseOk ){
	cout<<"--------- xy->xDense"<<prettyDims(xy->xDense)<<":\n";
	if(xy->xDense.size()<1000&&verbose>=3) cout<<xy->xDense<<endl;
      }else{ //xy->sparseOk
	cout<<"--------- xy->xSparse"<<prettyDims(xy->xSparse)<<":\n";
	if(xy->xSparse.size()<1000&&verbose>=3) cout<<xy->xSparse<<endl;
      }
    }
  }  

  */

  
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
			   
