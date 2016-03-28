
#include "mcprojProg.hpp"
#include "printing.hh"
#include "normalize.h"
#include "mcpredict.hh"
#include "predict.hh"           // low-level tests [optional]
#include "utils.h"              // OUTWIDE

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
    MCprojProgram::MCprojProgram( int argc, char**argv
                                  , int const verbose/*=0*/ )
        : ::opt::MCprojArgs( argc, argv )
        //, ::MCsolver( solnFile.size()? solnFile.c_str(): (char const* const)nullptr )
        , xDense()
        , denseOk(false)
        , xSparse()
        , sparseOk(false)
        , y() // SparseMb
    {
        // defaults: if given explicitly as \c defparms
        if(verbose>=1){
            //++A::verbose;
            cout<<" +MCprojProgram --xfile="<<A::xFile <<" --yFile="<<A::yFile;
            if(A::solnFile.size()) cout<<" --solnFile="<<A::solnFile;
            if(A::outFile.size()) cout<<" --output="<<A::outFile;
            if(A::outBinary) cout<<" -B";
            if(A::outText)   cout<<" -T";
            if(A::outSparse) cout<<" -S";
            if(A::outDense)  cout<<" -D";
            if(A::yFile.size()) cout<<" --yfile="<<A::yFile;
            //if(A::threads)   cout<<" --threads="<<A::threads;
            if(A::xnorm)     cout<<" --xnorm";
            cout<<endl;
        }
    }

    void MCprojProgram::tryRead( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1){cout<<" MCprojProgram::tryRead()"; cout.flush();}
        //MCsoln soln;
        {
            try{
                ifstream sfs(A::solnFile);
                if(!sfs.good()) {cerr<<" solnfile "<<A::solnFile<<endl; throw std::runtime_error("trouble opening soln file");}
                soln.read( sfs );
                sfs.close();
            }catch(std::exception const& e){
                cerr<<"Error reading solnfile "<<A::solnFile<<endl<<e.what()<<endl;
                throw;
            }
        }
        if(verbose>=2) soln.pretty(cout);
        // read the following MCprojProgram data:
        //DenseM xDense;
        //bool denseOk=false;
        //SparseM xSparse;
        //bool sparseOk=false;
        {
            ifstream xfs;
            // TODO XXX try Dense-Text, Sparse-Text too?
            try{
                if(verbose>=2) cout<<" try reading DenseM from xFile="<<xFile<<endl;
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
#if 0 // sparse projection seems ok now...
        if(xnorm && sparseOk){
            xDense = DenseM( xSparse );     // convert sparse --> dense
            xSparse.resize(0,0);
            sparseOk = false;
            denseOk = true;
        }
#endif
        //SparseMb y;                         // now this is optional
        // read SparseMb y;
        if(A::yFile.size()){
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
                //throw;
            }catch(std::exception const& e){
                cerr<<"ERROR: during read of classes from "<<yFile<<" -- "<<e.what()<<endl;
                throw;
            }
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
            assert( y.size() > 0 );
        }else{
            assert( y.size() == 0 );
        }
        assert( denseOk || sparseOk );
        if(verbose>=2){
            if( denseOk ){
                cout<<"--------- xDense"<<prettyDims(xDense)<<":\n";
                if(xDense.size()<1000||verbose>=3) cout<<xDense<<endl;
            }else{ //sparseOk
                cout<<"--------- xSparse"<<prettyDims(xSparse)<<":\n";
                if(xSparse.size()<1000||verbose>=3) cout<<xSparse<<endl;
                //if( xnorm ){ cout<<" xnorm!"<<endl; column_normalize(xSparse,xmean,xstdev); }
                // col-norm DISALLOWED
                if( xnorm ) throw std::runtime_error("sparse --xfile does not support --xnorm");
            }
            if( y.size() <= 0 ) cout<<"y: (no validation data)"<<endl;
            else{
                cout<<"--------- y"<<prettyDims(y)<<":\n";
                if(y.size()<1000||verbose>=3) cout<<y<<endl;
            }
        }
        //throw std::runtime_error("TBD");
    }
    void MCprojProgram::tryProj( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1) cout<<" MCprojProgram::tryProj() "<<(denseOk?"dense":sparseOk?"sparse":"HUH?")<<endl;
#if 0
        if(1){ // test that one of the low-level routines runs...
            cout<<"\t\tjust for show test... START (deprecated)"<<endl;
            VectorXsz no_active;
            ActiveDataSet * ads=nullptr;
            if( denseOk ){
                ads = getactive( no_active, xDense,
                                 soln.weights_avg, soln.lower_bounds_avg, soln.upper_bounds_avg,
                                 /*verbose=*/true );
            }else{
                ads = getactive( no_active, xSparse,
                                 soln.weights_avg, soln.lower_bounds_avg, soln.upper_bounds_avg,
                                 /*verbose=*/true );
            }
            assert( ads != nullptr );
            free_ActiveDataSet(ads); delete ads; ads=nullptr;
            cout<<"Good: got back from original 'getactive' call"<<endl;
            cout<<"\t\tjust for show test... DONE"<<endl;
        }
#endif
        if( denseOk ){
            assert( y.size()==0 || xDense.rows() == y.rows() );
            if( A::xnorm ){
                if(verbose>=2) cout<<" --xnorm doing col-norm of x data"<<endl;
                VectorXd xmean;
                VectorXd xstdev;
                if(verbose>=3){
                    cout<<"xDense ORIG:\n"<<xDense<<endl;
                    cout<<" xnorm!"<<endl;
                }
#if 1
                column_normalize(xDense,xmean,xstdev);
                if(verbose>=3){
                    cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
                    cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
                }
#else
                normalize_col(xDense);
#endif
            }
            feasible = project( xDense, soln );
            if(verbose>=1){                                // dump x and 'feasible' classes
                cout<<" feasible["<<feasible.size()<<"] classes, after project(xDense,soln)"; cout.flush();
                if(verbose>=2){
                    cout<<":"<<endl;
                    for(uint32_t i=0U; i<feasible.size(); ++i){
                        auto const& fi = feasible[i];
                        cout<<" x.row("<<setw(4)<<i<<")=";
                        OUTWIDE(cout,40,xDense.row(i));
                        cout<<"   classes: ";
                        for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) cout<<" "<<c;
                        //for(uint32_t c=0U; c<fi.size(); ++c) cout<<fi[c];
                        cout<<endl;
                    }
                }else{
                    cout<<endl;
                }
            }
        }else if( sparseOk ){
            if(A::xnorm && verbose>=1) cout<<"// normalization NOT YET SUPPORTED for sparse"<<endl;
            feasible = project( xSparse, soln );
            if(verbose>=1){                                    // dump x and 'feasible' classes
                cout<<" feasible["<<feasible.size()<<"] classes, after project(xDense,soln)"; cout.flush();
                if(verbose>=2){
                    cout<<":"<<endl;
                    for(uint32_t i=0U; i<feasible.size(); ++i){
                        auto const& fi = feasible[i];
                        cout<<" x.row("<<setw(4)<<i<<")=";
                        //OUTWIDE(cout,40,xSparse.row(i)); // Eigen bug: unwanted CR (dense row output OK)
                        {
                            ostringstream oss;
                            oss<<xSparse.row(i);
                            string s = oss.str();
                            cout<<setw(40)<<s.substr(0,s.size()-1);
                        }
                        cout<<"   classes: ";
                        for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) cout<<" "<<c;
                        cout<<endl;
                    }
                }else{
                    cout<<endl;
                }
            }
        }else{
            throw std::runtime_error("neither sparse nor dense training x was available");
        }
    }
    static void dumpFeasible(std::ostream& os, std::vector<boost::dynamic_bitset<>> const& vbs
                             , bool denseFmt=false, size_t firstRows=0U)
    {
        size_t const nOut = std::min( vbs.size(), (firstRows?firstRows:vbs.size()));
        for(uint32_t i=0U; i<nOut; ++i){
            auto const& fi = vbs[i];
            os<<i<<" ";
            if( denseFmt ){
                for(uint32_t c=0U; c<fi.size(); ++c) os<<fi[c];
            }else{
                for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) os<<" "<<c;
            }
            os<<endl;
        }
    }
    void MCprojProgram::trySave( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1) cout<<" MCprojProgram::trySave()"
            <<"\tSaving feasible "<<(outBinary?"Binary":"Text")<<" "
                <<(outSparse?"Sparse":"Dense")<<" classes --> "
                <<(outFile.size()? outFile: string("cout"))
                <<endl;
        if(outFile.size()==0){ // to cout, as Text
            if(verbose>=0 && outBinary) cout<<"## Warning: Binary flag ignored for cout"<<endl;
            cout<<"## solnFile: "<<A::solnFile<<endl;
            cout<<"##    xFile: "<<A::xFile<<endl;
            cout<<"## feasible["<<feasible.size()<<"x"<<soln.lower_bounds.rows()<<"] classes"
                ", after project(x,soln):"<<endl;
            dumpFeasible( cout, feasible, A::outDense, /*firstRows=*/0U );
        }else{
            ofstream ofs(outFile);
            if( ! ofs.good() ) throw std::runtime_error("trouble opening outFile");
            if(outBinary){
                if(outSparse){
                    cout<<"## Warning: Sparse flag ignored -- see printing.hh to implement it 'magically'"<<endl;
                }
                detail::io_bin( ofs, feasible );
                if(verbose>=2) cout<<"Note:\tIn C++, use printing.hh\n\t\tio_bin(ifstream(\""<<outFile
                    <<"\"),vector<boost::dynamic_bitset<>>&)\n\tcan read the projections binary file";
            }else{ // outText
                ofs<<"## solnFile: "<<A::solnFile<<endl;
                ofs<<"##    xFile: "<<A::xFile<<endl;
                ofs<<"## feasible["<<feasible.size()<<"x"<<soln.lower_bounds.rows()<<"] classes"
                ", after project(x,soln):"<<endl;
                dumpFeasible( ofs, feasible, A::outDense, /*firstRows=*/0U );
            }
            if( ! ofs.good() ) throw std::runtime_error("trouble writing outFile");
            ofs.close();
        }
    }
    void MCprojProgram::tryValidate( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if( y.size() ){
            if(verbose>=0) cout<<" McprojProgram::tryValidate()  "
                <<" Warning: validation against --yfile "<<yFile<<" TBD"<<endl;
        }else{
            if(verbose>=1) cout<<" MCprojProgram::tryValidate()  "
                <<endl;
        }
    }
}//opt::
