
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
    }
    void MCsolveProgram::tryRead( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::tryRead()"<<endl;
        // read the following MCsolveProgram data:
        //DenseM xDense;
        //bool denseOk=false;
        //SparseM xSparse;
        //bool sparseOk=false;
        {
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
        {
            ifstream yfs;
            try{
                yfs.open(yFile);
                if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
                ::detail::eigen_io_binbool( yfs, y );
                assert( y.cols() > 0U );
                if( yfs.fail() ) throw std::underflow_error("problem reading yfile with eigen_io_binbool");
            }catch(po::error& e){
                cerr<<"Invalid argument: "<<e.what()<<endl;
                throw;
            }catch(std::runtime_error const& e){
                cerr<<e.what()<<endl;
                throw;
            }catch(std::exception const& e){
                cerr<<"ERROR: during read of classes from "<<yFile<<" -- "<<e.what()<<endl;
                throw;
            }
        }
        assert( denseOk || sparseOk );
        if(verbose>=1){
            if( denseOk ){
                cout<<"xDense:\n"<<xDense<<endl;
                cout<<"y:\n"<<y<<endl;
                cout<<"parms:\n"<<A::parms<<endl;
                assert( xDense.rows() == y.rows() );
            }else{ //sparseOk
                cout<<"xSparse:\n"<<xSparse<<endl;
                cout<<"y:\n"<<y<<endl;
                cout<<"parms:\n"<<A::parms<<endl;
                //if( xnorm ){ cout<<" xnorm!"<<endl; column_normalize(xSparse,xmean,xstdev); }
                // col-norm DISALLOWED
                if( xnorm ) throw std::runtime_error("sparse --xfile does not support --xnorm");
                assert( xSparse.rows() == y.rows() );
            }
        }
        //throw std::runtime_error("TBD");
    }
    void MCsolveProgram::trySolve( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::trySolve() "<<(denseOk?"dense":sparseOk?"sparse":"HUH?")<<endl;
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
            S::solve( xDense, y, &(A::parms) );
        }else if( sparseOk ){
            // normalization NOT YET SUPPORTED for sparse
            S::solve( xSparse, y, &(A::parms) );
        }else{
            throw std::runtime_error("neither sparse nor dense training x was available");
        }
        // S::solve uses A::parms for the run, and will update S:parms to record
        // how the next outFile (.soln) was obtained.
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
    void MCsolveProgram::tryDisplay( int const verb/*=0*/ ){
        int const verbose = A::verbose + verb;
        if(verbose>=1) cout<<"MCsolveProgram::tryRead()"<<endl;
        MCsoln & soln = S::getSoln();
        DenseM      & w = soln.weights_avg;
        DenseM      & ww = soln.weights;
        cout<<" weights     norms: "; for(uint32_t c=0U; c<ww.rows(); ++c){cout<<" "<<ww.col(c).norm();} cout<<endl;
        cout<<" weights_avg norms: "; for(uint32_t c=0U; c<w.rows(); ++c){cout<<" "<<w.col(c).norm();} cout<<endl;
        w.colwise().normalize();                     // modify w, ww to unit-vector (? largest coeff +ve ?)
        ww.colwise().normalize();
        DenseM const& l = soln.lower_bounds_avg;
        DenseM const& u = soln.upper_bounds_avg;
        if(verbose>=1){
            cout<<"normalized     weights"<<prettyDims(ww)<<":\n"<<ww<<endl;
            cout<<"normalized weights_avg"<<prettyDims(w)<<":\n"<<w<<endl;
            cout<<"      lower_bounds_avg"<<prettyDims(l)<<":\n"<<l<<endl;
            cout<<"      upper_bounds_avg"<<prettyDims(u)<<":\n"<<u<<endl;
        }
        if(1){
            for(int p=0U; p<w.cols(); ++p){   // for each projection
                cout<<" Projection "<<p<<" weights "<< w.col(1).transpose();
                uint32_t c=0U;
                for(uint32_t c=0U; c<l.rows(); ++c){ // for each class
                    if( c%8U == 0U ) {cout<<"\n {l,u}:"<<setw(4)<<c;}
                    cout<<" { "<<setw(9)<<l.coeff(c,p)<<","<<setw(9)<<u.coeff(c,p)<<"}";
                }
                if(c%8U==0U) cout<<endl;
            }
        }
    }
}//opt::
