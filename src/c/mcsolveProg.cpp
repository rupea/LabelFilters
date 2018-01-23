#include "mcsolveProg.hpp"
#include "mcprojProg.hpp"
#include "mcxydata.h"
#include "printing.hh"
#include "normalize.h"

//#include <omp.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace opt {
    using namespace std;
    /**
     * - What are the default parameters?
     *   - 1. if given explicitly as \c defparms
     *   - 2. if --solnFile, adopt parms from a previous run
     *   - 3. else use library defaults
     */
  MCsolveProgram::MCsolveProgram( int argc, char**argv
				  , int const verbose/*=0*/
				  , param_struct const* const defparms/*=nullptr*/ )
    : ::opt::MCsolveArgs( argc, argv )
    , ::MCsolver() // solnFile.size()? solnFile.c_str(): (char const* const)nullptr )
    , projProg()                // empty shared ptr
    , xy( new ::MCxyData() )
  {
    // defaults: if given explicitly as \c defparms
    if( defparms ){
      A::parms = *defparms;
      parse(argc,argv);
    }    
    if( solnFile.size() ){
      ifstream ifs(solnFile.c_str());
      if( ifs.good() ) try{
	  if (A::parms.verbose >= 1 ) 
	    cout<<" reading "<<solnFile<<endl;
	  S::read( ifs );
	  if (A::parms.verbose >= 1 ) 
	    {
	      S::pretty( cout, A::parms.verbose );
	      cout<<" Done reading "<<solnFile<<endl;
	    }
	}catch(std::exception const& e){
	  ostringstream err;
	  err<<"ERROR: unrecoverable error reading MCsoln from file "<<solnFile << endl;
	  throw(runtime_error(err.str()));
	}
      // if (A::parms.verbose >= 1 ) 
      // 	cout<<" Using parameters from "<<solnFile<<" as defaults"<<endl;
      // A::parms = S::getParms();   // solnFile parms provide new "default" settings
      // parse(argc,argv);           // so must parse args again for cmdline overrides
    }else{
      if( A::parms.resume ) throw std::runtime_error(" --resume needs --solnfile=... or explicit default parms");
      if( A::parms.reoptimize_LU ) throw std::runtime_error(" --reoptlu needs --solnfile=... or explicit default parms");
    }
    if(A::parms.verbose>=1){
      cout<<" +MCsolveProgram --xfile="<<A::xFile <<" --yFile="<<A::yFile;
      if(A::solnFile.size()) cout<<" --solnFile="<<A::solnFile;
      if(A::outFile.size()) cout<<" --output="<<A::outFile;
      if(A::outBinary) cout<<" -B";
      if(A::outText)   cout<<" -T";
      if(A::yFile.size()) cout<<" --yfile="<<A::yFile;
      //if(A::threads)   cout<<" --threads="<<A::threads;
      if(A::xnorm)     cout<<" --xnorm";
      if(A::xunit)     cout<<" --xunit";
      if(A::xscale!=1.0) cout<<" --xscale="<<A::xscale;
      cout<<endl;
    }
  }
  MCsolveProgram::~MCsolveProgram(){
    if( projProg && projProg->solveProg ){  // reset known shared_ptr to me [circular]
      //projProg->solveProg.reset();
      projProg->solveProg = nullptr;
    }
  }
  void MCsolveProgram::tryRead( int const verb/*=0*/ ){
    int const verbose = A::verbose + verb;
    if(verbose>=1) cout<<"MCsolveProgram::tryRead()"<<endl;
    // read the following MCsolveProgram data:
    //DenseM  xDense;
    //bool    denseOk=false;
    //SparseM xSparse;
    //bool    sparseOk=false;
    if( yFile.size() == 0 && xFile.size() != 0 ){
      try{
	ifstream xfs;
	// try to read a text libsvm format -- xy->y labels, then SparseX data
	xfs.open(xFile);
	if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
	detail::eigen_read_libsvm( xfs, xy->xSparse, xy->y );
	xy->sparseOk = true;
	xfs.close();
      }catch(std::exception const& e){
	cerr<<" --xFile="<<xFile<<" and no --yFile: libsvm format input error!"<<endl;
	throw;
      }
    }
    if(!xy->sparseOk){
      xy->xread(xFile);
    }
    if(xnorm && xy->sparseOk){
      xy->xDense = DenseM( xy->xSparse );     // convert sparse --> dense
      xy->xSparse.resize(0,0);
      xy->sparseOk = false;
      xy->denseOk = true;
    }
    // read SparseMb xy->y;
    if(yFile.size()){
      xy->yread(yFile);
      if( !(xy->y.size() > 0) ) throw std::runtime_error("trouble reading y data");
    }
#ifndef NDEBUG
    assert( xy->denseOk || xy->sparseOk );
    if( xy->denseOk ){
      assert( xy->xDense.rows() == xy->y.rows() );
    }else{ //xy->sparseOk
      assert( xy->xSparse.rows() == xy->y.rows() );
      // col-norm DISALLOWED
    }
#endif
    if( A::xnorm ){
      if (!xy->denseOk)	throw std::runtime_error("sparse --xfile does not support --xnorm");      
      ::Eigen::VectorXd xmean;
      ::Eigen::VectorXd xstdev;
      if(verbose>=1){
	cout<<" xnorm!"<<endl;
      }
      xy->xstdnormal(xmean, xstdev, true, true, false);
      if(verbose>=1){
	cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
	cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
      }
    }
    
    if( A::xunit ){
      xy->xunitnormal();
    }
    if( A::xscale != 1.0 ){
      xy -> xscale( A::xscale );
    }
    
    if(verbose>=1){
      if( xy->denseOk ){
	cout<<"--------- xy->xDense"<<prettyDims(xy->xDense)<<":\n";
	if(xy->xDense.size()<1000||verbose>=3) cout<<xy->xDense<<endl;
      }else{ //xy->sparseOk
	cout<<"--------- xy->xSparse"<<prettyDims(xy->xSparse)<<":\n";
	if(xy->xSparse.size()<1000||verbose>=3) cout<<xy->xSparse<<endl;
	//if( xnorm ){ cout<<" xnorm!"<<endl; col_normalize(xy->xSparse,xmean,xstdev); }
	// col-norm DISALLOWED
	if( xnorm ) throw std::runtime_error("sparse --xfile does not support --xnorm");
      }
      if( xy->y.size() <= 0 ) cout<<"xy->y: (no validation data)"<<endl;
      else{
	cout<<"--------- xy->y"<<prettyDims(xy->y)<<":\n";
	if(xy->y.size()<1000||verbose>=3) cout<<xy->y<<endl;
      }
    }
  }
  /** \sa mcsolver.hh for \ref MCsolver::solve implementation.
   * \ref MCsolver::solve is a compact rewrite of the original
   * \ref solve_optimization routine. */
  void MCsolveProgram::trySolve( int const verb/*=0*/ ){
    int const verbose = A::verbose + verb;
    if(verbose>=1) cout<<"MCsolveProgram::trySolve() "<<(xy->denseOk?"dense":xy->sparseOk?"sparse":"HUH?")<<endl;
    if( xy->denseOk ){
      S::solve( xy->xDense, xy->y, &(A::parms) );
    }else if( xy->sparseOk ){
      // normalization NOT YET SUPPORTED for sparse
      S::solve( xy->xSparse, xy->y, &(A::parms) );
    }else{
      throw std::runtime_error("neither sparse nor dense training x was available");
    }
    // S::solve uses A::parms for the run, and will update S:parms to record
    // how the next outFile (.soln) was obtained.
#if 0
    // --- post-processing --- opportunity to add 'easy' stuff to MCsoln ---
    if( xy->denseOk ){
      S::setQuantiles( xy->xDense, xy->y );
    }else if( xy->sparseOk ){
      S::setQuantiles( xy->xSparse, xy->y );
    }
    // ---------------------------------------------------------------------
#endif
  }
  void MCsolveProgram::trySave( int const verb/*=0*/ ){
    int const verbose = A::verbose + verb;
    if(verbose>=1) cout<<"MCsolveProgram::trySave()"<<endl;
    MCsoln & soln = S::getSoln();
    if( A::outFile.size() ){
      if(verbose>=1){
	cout<<" Writing MCsoln";
	if( A::solnFile.size() ) cout<<" initially from "<<A::solnFile;
	cout<<" to "<<A::outFile<<endl;
      }
      { // should I have an option to print the soln to cout? NAAH
	ofstream ofs;
	try{
	  ofs.open(A::outFile);
	  if( ! ofs.good() ) throw std::runtime_error("trouble opening outFile");
	  soln.write( ofs, (A::outBinary? MCsoln::BINARY: MCsoln::TEXT));
	  if( ! ofs.good() ) throw std::runtime_error("trouble writing outFile");
	  ofs.close();
	}catch(std::exception const& e){
	  cerr<<" trouble writing "<<A::outFile<<" : "<<e.what()<<endl;
	  ofs.close();
	  throw;
	}catch(...){
	  cerr<<" trouble writing "<<A::outFile<<" : unknown exception"<<endl;
	  ofs.close();
	  throw;
	}
	if(verbose>=1) cout<<"\tmcdumpsoln -p < "<<A::outFile<<" | less    # to prettyprint the soln"<<endl;
      }
    }
  }
  void MCsolveProgram::savex( std::string fname ) const {
    xy-> xwrite( fname ); // handles "current" x format (sparse/dense)
  }
  void MCsolveProgram::savey( std::string fname ) const {
    xy-> ywrite( fname ); // only binary save here, inefficient :(
  }
  
  /** transform EVERY row of x by adding the "quadratic dimensions".1,
   * i.e. append the data values of the outer product of the row vectors.
   * Should use full parallelism for this! \throw if no data.
   * \sa MCxyData::quadx(). */
  void MCsolveProgram::quadx() {
    xy->quadx();                    // autoscale
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
      //cout<<"."; cout.flush();
      if( narrow.size() < maxNarrow
	  || width <= u.coeff(narrow.back(),p)-l.coeff(narrow.back(),p))
	{
	  //cout<<" c="<<c<<" wid:"<<width;
	  if( narrow.size() >= maxNarrow )
	    narrow.pop_back();
#if 1 // WORKS
	  size_t big=0U;
	  for( ; big<narrow.size(); ++big ){
	    if( u.coeff(narrow[big],p)-l.coeff(narrow[big],p) > width )
	      break;
	  }
	  //cout<<" big="<<big;
	  if( big < narrow.size() ){
	    narrow.push_back(0); // likely in wrong postion
	    for(size_t b=narrow.size()-1U; b>big; --b)
	      narrow[b] = narrow[b-1];
	    narrow[big] = c;
	  }else{
	    narrow.push_back(c);
	  }
#else // problems with STL way ...
	  auto iter = lower_bound(narrow.begin(),narrow.end(), width,
				  [narrow,u,l,p](size_t const& a, double w) {
				    size_t const na = narrow[a];
				    double const wa = u.coeff(na,p) - l.coeff(na,p);
				    return wa < w;
				  });
	  {
	    cout<<" narrow: ";
	    for(size_t i=0U; i<narrow.size(); ++i){
	      size_t cls=narrow[i];
	      cout<<setw(6)<<cls<<" {"<<setw(10)<<l.coeff(cls,p)
		  <<", "<<u.coeff(cls,p)<<"}"<<u.coeff(cls,p)-l.coeff(cls,p);
	      //cout<<endl;
	    }
	    cout<<endl;
	  }
	  narrow.insert(iter,c); // insert before 'iter'
#endif
	  if(0){
	    cout<<"--> narrow: ";
	    for(size_t i=0U; i<narrow.size(); ++i){
	      size_t cls=narrow[i];
	      cout<<setw(6)<<cls<<" {"<<setw(10)<<l.coeff(cls,p)
		  <<", "<<u.coeff(cls,p)<<"}"<<u.coeff(cls,p)-l.coeff(cls,p);
	      //cout<<endl;
	    }
	    cout<<endl;
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
                //if(verbose>=2) cout<<" c="<<c<<" wid:"<<width<<" Xbig="<<big;
                --big;                    // copy elements towards wide.begin();
                for(size_t b=0; b<big; ++b) wide[b] = wide[b+1];
            }else{ // insert-before, with growing
                //if(verbose>=2) cout<<" c="<<c<<" wid:"<<width<<" big="<<big;
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
                    //cout<<endl;
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

    void MCsolveProgram::tryDisplay( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::tryDisplay()"<<endl;
        MCsoln const& soln = S::getSoln();
        // NOTE: for normalization make a COPY (can be avoided) FIXME
        DenseM const& w = soln.weights;
        DenseM const& l = soln.lower_bounds;
        DenseM const& u = soln.upper_bounds;
        if(verbose>=1){ // really want to find a nicer, compact display here XXX
            cout<<"normalized weights"<<prettyDims(w)<<":\n";
            if( w.size() > 0U && w.size() < 500U ){
                vector<double>  wNorms(w.cols());
                for(uint32_t c=0U; c< w.cols(); ++c)
                    wNorms[c]=  w.col(c).norm();
                cout<<" weights norms: ";for(uint32_t c=0U; c<w.cols(); ++c){cout<<" "<<wNorms[c];}cout<<endl;
                for(uint32_t c=0U; c<w.cols(); ++c)
                    cout<<w.col(c)*(wNorms[c]>1.e-8?1.0/wNorms[c]:1.0)<<endl;
            }
            cout<<"      lower_bounds"<<prettyDims(l)<<":\n";
            if( l.size() < 500U ) cout<<l<<endl;
            cout<<"      upper_bounds"<<prettyDims(u)<<":\n";
            if( u.size() < 500U ) cout<<u<<endl;
        }
        if(1){
            for(int p=0U; p<w.cols(); ++p){   // for each projection
                cout<<" Projection "<<p<<" weights[ "<<w.rows()<<"] ";
                if( w.rows()<20U ) cout<< w.col(1).transpose();
                uint32_t c=0U;
                for(uint32_t c=0U; c<std::min((uint32_t)l.rows(),uint32_t{64U}); ++c){ // for each class
                    if( c%8U == 0U ) {cout<<"\n {l,u}:"<<setw(4)<<c;}
                    cout<<" { "<<setw(9)<<l.coeff(c,p)<<","<<setw(9)<<u.coeff(c,p)<<"}";
                }
                cout<<" ...";
                if(c%8U==0U) cout<<endl;
                size_t wrong = printNarrowIntervals( cout, /*maxNarrow=*/10U, l, u, p );
                /*wrong =*/ printWideIntervals( cout, /*maxWide=*/4U, l, u, p );
                cout<<" "<<wrong<<" classes had vanishing intervals, with lower > upper."<<endl;
                if(wrong) cout<<" To help allow some of these "<<wrong<<" classes to be found,\n"
                    <<" consider running with higher C1 / lower C2"<<endl;
            }
        }
        if(1) { // NEW TEST CODE XXX
            cout<<"\n********* NEW TEST CODE **********"<<endl;
            this->pretty(cout, verbose);
            auto proj = projector("-v");
            if( ! proj ) throw std::runtime_error(" Failed to construct projector associated with this solver");

            //proj->tryRead(7/*verbose*/);
            //cout<<"tryRead MCprojProgram soln is "; proj->soln.pretty(cout); cout<<endl;

            proj->tryProj(7/*verbose*/);
            proj->tryValidate(7/*verbose*/);
        }
    }

    //std::shared_ptr<MCprojProgram> MCsolveProgram::projector(std::string args)
    MCprojProgram * MCsolveProgram::projector(std::string args)
    {
        if( ! projProg ){
            std::ostringstream cmd;            // default command string
            // Potential difficulty -- embedded whitespace will be treated INCORRECTLY
            cmd<<" INTERNAL --solnfile="<<(A::outFile.size()?A::outFile:A::solnFile)
                <<" --xfile="<<xFile<<" --yfile="<<yFile
                <<" "<<args;
            //projProg = std::make_shared<MCprojProgram>( this, cmd.str() );
            projProg = new MCprojProgram( this, cmd.str() );
            return projProg;
        }
        return projProg;
    }
}//opt::
