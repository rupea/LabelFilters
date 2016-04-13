
#include "mcsolveProg.hpp"
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
            , ::MCsolver( solnFile.size()? solnFile.c_str(): (char const* const)nullptr )
                , xDense()
                , denseOk(false)
                , xSparse()
                , sparseOk(false)
                , y() // SparseMb
    {
        // defaults: if given explicitly as \c defparms
        if( defparms ){
            cout<<" reparse cmdline, this time with supplied defaults"<<endl;
            A::parms = *defparms;
            parse(argc,argv);
        }else if( solnFile.size() ){
                cout<<" reparse cmdline, this time with "<<solnFile<<" parms as defaults"<<endl;
                A::parms = S::getParms();   // solnFile parms provide new "default" settings
                parse(argc,argv);           // so must parse args again for cmdline overrides
        }else{
            if( A::parms.resume ) throw std::runtime_error(" --resume needs --solnfile=... or explicit default parms");
            if( A::parms.reoptimize_LU ) throw std::runtime_error(" --reoptlu needs --solnfile=... or explicit default parms");
        }
        if(verbose>=1){
            cout<<" +MCsolveProgram --xfile="<<A::xFile <<" --yFile="<<A::yFile;
            if(A::solnFile.size()) cout<<" --solnFile="<<A::solnFile;
            if(A::outFile.size()) cout<<" --output="<<A::outFile;
            if(A::outBinary) cout<<" -B";
            if(A::outText)   cout<<" -T";
            if(A::outShort)  cout<<" -S";
            if(A::outLong)   cout<<" -L";
            if(A::yFile.size()) cout<<" --yfile="<<A::yFile;
            //if(A::threads)   cout<<" --threads="<<A::threads;
            if(A::xnorm)     cout<<" --xnorm";
            if(A::xunit)     cout<<" --xunit";
            if(A::xscale!=1.0) cout<<" --xscale="<<A::xscale;
            cout<<endl;
        }
    }
    void MCsolveProgram::tryRead( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::tryRead()"<<endl;
        // read the following MCsolveProgram data:
        //DenseM xDense;
        //bool denseOk=false;
        //SparseM xSparse;
        //bool sparseOk=false;
        if( yFile.size() == 0 && xFile.size() != 0 ){
            try{
                ifstream xfs;
                // try to read a text libsvm format -- Y labels, then SparseX data
                xfs.open(xFile);
                if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
                detail::eigen_read_libsvm( xfs, xSparse, y );
                sparseOk = true;
                xfs.close();
            }catch(std::exception const& e){
                cerr<<" --xFile="<<xFile<<" and no --yFile: libsvm format input error!"<<endl;
                throw;
            }
        }
        if(!sparseOk){
            ifstream xfs;
            // TODO XXX try Dense-Text, Sparse-Text too?
            try{
                xfs.open(xFile);
                if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
                ::detail::eigen_io_bin(xfs, xDense);
                if( xfs.fail() ) throw std::underflow_error("problem reading DenseM from xfile with eigen_io_bin");
                char c;
                xfs >> c;   // should trigger eof if BINARY dense file exactly the write length
                if( ! xfs.eof() ) throw std::overflow_error("xDense read did not use full file");
                xfs.close();
                assert( xDense.cols() > 0U );
                denseOk=true;
            }catch(po::error& e){
                cerr<<"Invalid argument: "<<e.what()<<endl;
                throw;
            }catch(std::exception const& what){
                cerr<<"Retrying xFile as SparseM..."<<endl;
                try{
                    xfs.close();
                    xfs.open(xFile);
                    if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
                    ::detail::eigen_io_bin( xfs, xSparse );
                    if( xfs.fail() ) throw std::underflow_error("problem reading SparseM from xfile with eigen_io_bin");
                    xfs.close();
                    assert( xSparse.cols() > 0U );
                    sparseOk=true;
                }catch(...){
                    cerr<<" Doesn't seem to be sparse either"<<endl;
                }
            }
        }
        if(xnorm && sparseOk){
            xDense = DenseM( xSparse );     // convert sparse --> dense
            xSparse.resize(0,0);
            sparseOk = false;
            denseOk = true;
        }
        // read SparseMb y;
        if(yFile.size()){
            ifstream yfs;
            bool yOk = false;
            try{
                yfs.open(yFile);
                if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
                ::detail::eigen_io_binbool( yfs, y );
                assert( y.cols() > 0U );
                if( yfs.fail() ) throw std::underflow_error("problem reading yfile with eigen_io_binbool");
                yOk = true;
            }catch(po::error& e){
                cerr<<"Invalid argument: "<<e.what()<<endl;
                throw;
            }catch(std::runtime_error const& e){
                cerr<<"ERROR: tryRead(), "<<e.what()<<endl;
                //throw;
            }catch(std::exception const& e){
                cerr<<"ERROR: during read of classes from "<<yFile<<" -- "<<e.what()<<endl;
                throw;
            }
            if( !yOk ){
                cerr<<"Retrying --yfile as text mode list-of-classes format (eigen_io_txtbool)"<<endl;
                try{
                    yfs.close();
                    yfs.open(yFile);
                    if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
                    ::detail::eigen_io_txtbool( yfs, y );
                    assert( y.cols() > 0U );
                    // yfs.fail() is expected
                    if( ! yfs.eof() ) throw std::underflow_error("problem reading yfile with eigen_io_txtbool");
                }
                catch(std::exception const& e){
                    cerr<<" --file could not be read in text mode from "<<yFile<<" -- "<<e.what()<<endl;
                    throw;
                }
            }
            assert( y.size() > 0 );
        }
#ifndef NDEBUG
        assert( denseOk || sparseOk );
        if( denseOk ){
            assert( xDense.rows() == y.rows() );
        }else{ //sparseOk
            assert( xSparse.rows() == y.rows() );
            // col-norm DISALLOWED
        }
#endif
        if( sparseOk && xnorm )
            throw std::runtime_error("sparse --xfile does not support --xnorm");
        if( denseOk ){
            if( A::xnorm ){
                VectorXd xmean;
                VectorXd xstdev;
                if(verbose>=1){
                    cout<<"xDense ORIG:\n"<<xDense<<endl;
                    cout<<" xnorm!"<<endl;
                }
#if 1
                column_normalize(xDense,xmean,xstdev);
                if(verbose>=1){
                    cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
                    cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
                }
#else
                normalize_col(xDense);
#endif
            }
        }
        if( A::xunit ){
            VectorXd xSqNorms;
            if( denseOk ){
                for(size_t r=0U; r<xDense.rows(); ++r){
                    double const l2 = xDense.row(r).squaredNorm();
                    if( l2 > 1.e-10 ){
                        xDense.row(r) *= (1.0/sqrt(l2));
                    }
                }
            }else if( sparseOk ){
                for(size_t r=0U; r<xSparse.rows(); ++r){
                    double const l2 = xSparse.row(r).squaredNorm();
                    if( l2 > 1.e-10 ){
                        xSparse.row(r) *= (1.0/sqrt(l2));
                    }
                }
            }
        }
        if( A::xscale != 1.0 ){
            if( denseOk ){
                xDense *= A::xscale;
            }else if( sparseOk ){
                xSparse *= A::xscale;
            }
        }

        if(verbose>=1 && y.rows() < 50 ){       // print only for small tests
            if( denseOk ){
                cout<<"xDense:\n"<<xDense<<endl;
                cout<<"y:\n"<<y<<endl;
                cout<<"parms:\n"<<A::parms<<endl;
            }else{ //sparseOk
                cout<<"xSparse:\n"<<xSparse<<endl;
                cout<<"y:\n"<<y<<endl;
                cout<<"parms:\n"<<A::parms<<endl;
            }
        }
    }
    /** \sa mcsolver.hh for \ref MCsolver::solve implementation.
     * \ref MCsolver::solve is a compact rewrite of the original
     * \ref solve_optimization routine. */
    void MCsolveProgram::trySolve( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::trySolve() "<<(denseOk?"dense":sparseOk?"sparse":"HUH?")<<endl;
        if( denseOk ){
            S::solve( xDense, y, &(A::parms) );
        }else if( sparseOk ){
            // normalization NOT YET SUPPORTED for sparse
            S::solve( xSparse, y, &(A::parms) );
        }else{
            throw std::runtime_error("neither sparse nor dense training x was available");
        }
        // S::solve uses A::parms for the run, and will update S:parms to record
        // how the next outFile (.soln) was obtained.
#if 0
        // --- post-processing --- opportunity to add 'easy' stuff to MCsoln ---
        if( denseOk ){
            S::setQuantiles( xDense, y );
        }else if( sparseOk ){
            S::setQuantiles( xSparse, y );
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
            soln.fname = A::solnFile;
            { // should I have an option to print the soln to cout? NAAH
                ofstream ofs;
                try{
                    ofs.open(A::outFile);
                    if( ! ofs.good() ) throw std::runtime_error("trouble opening outFile");
                    soln.write( ofs, (A::outBinary? MCsoln::BINARY: MCsoln::TEXT)
                                ,    (A::outShort ? MCsoln::SHORT : MCsoln::LONG) );
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
        ofstream ofs;
        try{
            if( !sparseOk && !denseOk )
                throw std::runtime_error("savex no Eigen x data yet");
            ofs.open(fname);
            if( ! ofs.good() ) throw std::runtime_error("savex trouble opening fname");
            if( sparseOk ){
                detail::eigen_io_bin(ofs, xSparse);
            }else{ assert(denseOk);
                detail::eigen_io_bin(ofs, xDense);
            }
            if( ! ofs.good() ) throw std::runtime_error("savex trouble writing fname");
            ofs.close();
        }catch(std::exception const& e){
            cerr<<" trouble writing "<<A::outFile<<" : unknown exception"<<endl;
            ofs.close();
            throw;
        }
    }
    void MCsolveProgram::savey( std::string fname ) const {
        ofstream ofs;
        try{
            ofs.open(fname);
            if( ! ofs.good() ) throw std::runtime_error("savex trouble opening fname");
            detail::eigen_io_binbool(ofs, y);
            if( ! ofs.good() ) throw std::runtime_error("savex trouble writing fname");
            ofs.close();
        }catch(std::exception const& e){
            cerr<<" trouble writing "<<A::outFile<<" : unknown exception"<<endl;
            ofs.close();
            throw;
        }
    }
    
    /** convert x to new matrix adding quadratic dimensions to each InnerIterator (row).
     * Optionally scale the quadratic elements to keep them to reasonable range.
     * For example if original dimension is 0..N, perhaps scale the quadratic
     * terms by 1.0/N. */
    static void addQuadratic( SparseM & x, double const qscal=1.0 ){
        x.makeCompressed();
        VectorXi xsz(x.outerSize());
#pragma omp parallel for schedule(static)
        for(int i=0U; i<x.outerSize(); ++i){
            xsz[i] = x.outerIndexPtr()[i+1]-x.outerIndexPtr()[i];
        }
        // final inner dim increases by square of inner dim
        SparseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
        // calc exact nnz elements for each row of q
        VectorXi qsz(x.outerSize());
#pragma omp parallel for schedule(static)
        for(int i=0; i<x.outerSize(); ++i){
            qsz[i] = xsz[i] + xsz[i]*xsz[i];
        }
        q.reserve( qsz );           // reserve exact per-row space needed
        // fill q
#pragma omp parallel for schedule(dynamic,128)
        for(int r=0; r<x.outerSize(); ++r){
            for(SparseM::InnerIterator i(x,r); i; ++i){
                q.insert(r,i.col()) = i.value();    // copy the original dimension
                for(SparseM::InnerIterator j(x,r); j; ++j){
                    int col = x.innerSize() + i.col()*x.innerSize() + j.col(); // x.innerSize() is 4
                    q.insert(r,col) = i.value()*j.value()*qscal;  // fill in quad dims
                }
            }
        }
        q.makeCompressed();
        x.swap(q);
    }
    static void addQuadratic( DenseM & x, double const qscal=1.0 ){
        DenseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
#pragma omp parallel for schedule(static,128)
        for(int r=0; r<x.outerSize(); ++r){
            for(int i=0; i<x.innerSize(); ++i){
                q.coeffRef(r,i) = x.coeff(r,i);
                for(int j=0; j<x.innerSize(); ++j){
                    int col = x.innerSize() + i*x.innerSize() + j;
                    q.coeffRef(r,col) = x.coeff(r,i) * x.coeff(r,j) * qscal;
                }
            }
        }
        x.swap(q);
    }
    /** transform EVERY row of x by adding the "quadratic dimensions".
     * i.e. append the data values of the outer product of the row vectors. */
    void MCsolveProgram::quadx() {
        if( sparseOk ){
            addQuadratic( xSparse );    // transform the original data matrix
        }else if( denseOk ){
            addQuadratic( xDense );     // transform the original data matrix
        }else{
            throw std::runtime_error(" quadx needs some x data first");
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
        if(verbose>=1) cout<<"MCsolveProgram::tryRead()"<<endl;
        MCsoln & soln = S::getSoln();
        DenseM      & w = soln.weights_avg;
        DenseM      & ww = soln.weights;
        cout<<" weights     norms: "; for(uint32_t c=0U; c<ww.cols(); ++c){cout<<" "<<ww.col(c).norm();} cout<<endl;
        cout<<" weights_avg norms: "; for(uint32_t c=0U; c<w.cols(); ++c){cout<<" "<<w.col(c).norm();} cout<<endl;
        w.colwise().normalize();                     // modify w, ww to unit-vector (? largest coeff +ve ?)
        ww.colwise().normalize();
        DenseM const& l = soln.lower_bounds_avg;
        DenseM const& u = soln.upper_bounds_avg;
        if(verbose>=1){
            // really want to find a nicer, compact display here XXX
            cout<<"normalized     weights"<<prettyDims(ww)<<":\n";
            if( ww.size() < 500U ) cout<<ww<<endl;
            cout<<"normalized weights_avg"<<prettyDims(w)<<":\n";
            if( w.size() < 500U ) cout<<w<<endl;
            cout<<"      lower_bounds_avg"<<prettyDims(l)<<":\n";
            if( l.size() < 500U ) cout<<l<<endl;
            cout<<"      upper_bounds_avg"<<prettyDims(u)<<":\n";
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
    }
}//opt::
