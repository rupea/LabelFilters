#include "mclearnFilter.h"
#include "mcxydata.h"
#include "printing.hh"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

MClearnFilter::MClearnFilter( DP data, MCsoln const& soln /*=MCsoln()*/, SolverParams const& params /*=SolverParams()*/):
   MCsolver(soln) 
  , m_xy(data)
  , m_params(params)
{
  // check some parameter compatibility. Should not be done here since parameters can change.  
  if ( S::getSoln().nProj == 0 )
    {
      if (m_params.resume() )
	{
	  cerr << "WARNING: resume requested without providing a previous solution. Ignoring." << endl;
	  m_params.resume(false);
	}
      if (m_params.reoptimize_LU() )
	{
	  cerr << "WARNING: reoptimization of L, U requested without providing a previous solution. Ignoring." << endl;
	  m_params.reoptimize_LU(false);
	}
    }
  if (m_params.verbose()  >= 1)
    {
      cout << "Parameters: " << endl;
      cout << m_params << endl;
    }
}

MClearnFilter::~MClearnFilter(){}


void MClearnFilter::learn(){
  if(m_params.verbose() >= 1) cout<<"MClearnFilter::learn() "<<(m_xy->denseOk?"dense":m_xy->sparseOk?"sparse":"HUH?")<< " data" <<endl;
  if( m_xy->denseOk ){
    this->solve( m_xy->xDense, m_xy->y, m_params.params() );
  }else if( m_xy->sparseOk ){
    this->solve( m_xy->xSparse, m_xy->y, m_params.params() );
  }else{
    throw std::runtime_error("neither sparse nor dense training x was available");
  }
}

void MClearnFilter::saveSolution(std::string fname, MCsoln::Fmt fmt){

  if(m_params.verbose() >= 1) cout<<"MClearnFilter::saveSolution( " << fname << ", " <<  ((fmt == MCsoln::BINARY)?"binary":"text") << " )"<<endl;

  ofstream ofs;
  if (fname.size() > 0)
    {
      ofs.open(fname);
    }
  else
    {
      if (fmt == MCsoln::BINARY){
	cerr<<"Warning: Binary flag ignored for cout"<<endl;
	fmt = MCsoln::TEXT;
      }
    }
    
  ostream& out = fname.size()?ofs:cout;
  
  if( ! out.good() )
    {
      if (fname.size())
	{
	  ofs.close();
	}
      throw std::runtime_error("trouble opening the file");
    }
  
  S::write( out, fmt);

  if( ! out.good() )
    {
      if (fname.size())
	{
	  ofs.close();
	}
      throw std::runtime_error("trouble writing outFile");
    }
  
  if (fname.size())
    {
      ofs.close();
    }
}
  
/** print some smaller valid intervals, and return number of
 * classes with vanishing intervals. */
static size_t printNarrowIntervals( std::ostream&  os, size_t const maxNarrow,
				    DenseM const& l, DenseM const& u, size_t const p ){
  vector<size_t> narrow;
  size_t wrong=0U;
  for(size_t c=0U; c<l.rows(); ++c){
    if( l.coeff(c,p) > u.coeff(c,p) ){
      ++wrong;
      continue;
    }
    double width = u.coeff(c,p) - l.coeff(c,p);
    if( narrow.size() < maxNarrow
	|| width <= u.coeff(narrow.back(),p)-l.coeff(narrow.back(),p))
      {
	if( narrow.size() >= maxNarrow )
	  narrow.pop_back();
	size_t big=0U;
	for( ; big<narrow.size(); ++big ){
	  if( u.coeff(narrow[big],p)-l.coeff(narrow[big],p) > width )
	    break;
	}
	if( big < narrow.size() ){
	  narrow.push_back(0); // likely in wrong postion
	  for(size_t b=narrow.size()-1U; b>big; --b)
	    narrow[b] = narrow[b-1];
	  narrow[big] = c;
	}else{
	  narrow.push_back(c);
	}
      }
  }
  cout<<" Some narrow non-zero intervals were:"<<endl;
  for(size_t i=0U; i<narrow.size(); ++i){
    size_t cls=narrow[i];
    cout<<" class "<<setw(6)<<cls<<" {"<<setw(10)<<l.coeff(cls,p)<<", "
	<<u.coeff(cls,p)<<" } width "<<u.coeff(cls,p)-l.coeff(cls,p)<<endl;
  }
  return wrong;
}

static size_t printWideIntervals( std::ostream&  os, size_t const maxWide,
				  DenseM const& l, DenseM const& u, size_t const p ){
  int const verbose=0;
  vector<size_t> wide;
  size_t wrong=0U;
  for(size_t c=0U; c<l.rows(); ++c){
    if( l.coeff(c,p) > u.coeff(c,p) ){
      ++wrong;
      continue;
    }
    double width = u.coeff(c,p) - l.coeff(c,p);
    // put width into ascended sorted list wide[] of largest values
    size_t big=0U;                      // search for a wide[big]
    for( ; big<wide.size(); ++big )     // with (u-l) > width
      if( u.coeff(wide[big],p)-l.coeff(wide[big],p) > width )
	break;
    // shift entries of wide[] to make room for new entry (if nec.)
    if( wide.size() >= maxWide ){ // insert-before, without growing
      if( big == 0 ) continue;  // width not big enough to save
      --big;                    // copy elements towards wide.begin();
      for(size_t b=0; b<big; ++b) wide[b] = wide[b+1];
    }else{ // insert-before, with growing
      wide.push_back(0);       // copy elements towards wide.end()
      for(size_t b=wide.size()-1U; b>big; --b) wide[b] = wide[b-1];
    }
    // OK, shifty business is done. plop index 'c' into final resting place.
    wide[big] = c;
    
    if(verbose>=1){
      cout<<"--> wide: ";
      for(size_t i=0U; i<wide.size(); ++i){
	size_t cls=wide[i];
	cout<<setw(6)<<cls<<" {"<<setw(10)<<l.coeff(cls,p)
	    <<", "<<u.coeff(cls,p)<<"}"<<u.coeff(cls,p)-l.coeff(cls,p);
      }
      cout<<endl;
    }
  }
  cout<<" Some wide non-zero intervals were:"<<endl;
  for(size_t i=0U; i<wide.size(); ++i){
    size_t cls=wide[i];
    cout<<" class "<<setw(6)<<cls<<" {"<<setw(10)<<l.coeff(cls,p)<<", "
	<<u.coeff(cls,p)<<" } width "<<u.coeff(cls,p)-l.coeff(cls,p)<<endl;
  }
  return wrong;
}

void MClearnFilter::display(){
  if(m_params.verbose()>=1) cout<<"MClearnFilter::display()"<<endl;

  MCsoln const& soln = S::getSoln();
  DenseM const& w = soln.weights;
  DenseM const& l = soln.lower_bounds;
  DenseM const& u = soln.upper_bounds;
  cout<<"Filter weight matrix dimmension:"<<prettyDims(w)<<"\n";
  for(uint32_t c=0U; c< w.cols(); ++c)
    {
      cout<<"Filter "<< c << " :\n"
	  <<"   Norm: " << w.col(c).norm() << "\n";      
      size_t wrong = printNarrowIntervals( cout, /*maxNarrow=*/10U, l, u, c );
      /*wrong =*/ printWideIntervals( cout, /*maxWide=*/4U, l, u, c );
      cout<<" "<<wrong<<" classes had vanishing intervals, with lower > upper."<<endl;
      if(wrong) cout<<" To help allow some of these "<<wrong<<" classes to be found,\n"
		    <<" consider running with higher C1 / lower C2"<<endl;
    }
}
