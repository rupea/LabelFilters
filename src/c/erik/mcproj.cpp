/** \file
 * For every example in a test set, use <B>M</B>ulti-<B>C</B>lass <B>proj</B>ections
 * to determine {possible class labels}.
 */

#include "find_w.hh"
//#include "parameter-args.h"
#include "normalize.h"
#include "predict.hh"           // for testing
#include "mcpredict.hh"

#include <boost/program_options.hpp>

#include <string>
#include <fstream>

using namespace std;
using namespace boost::program_options;
namespace po = boost::program_options;

string xFile;           ///< x data file name (io via ???)
string solnFile;        ///< solution file basename

string yFile;           ///< y data file name (io via eigen_io_binbool)
bool xnorm=false;       ///< normalize x dims across examples to mean and stdev of 1.0
uint32_t threads;       ///< used?

void helpUsage( std::ostream& os ){
    os  <<" Function:"
        <<"\n    apply .soln {w,l,u} projections to examples 'x' and print eligible class"
        <<"\n    assignments of each example (row of 'x')"
        <<"\n Usage:"
        <<"\n    mcproj --xfile=... --solnfile=... [other args...]"
        <<"\n xfile is a plain eigen DenseM or SparseM (always stored as float)"
        <<"\n soln is a .soln file, such as output by mcgen or mcsolve"
        <<endl;
}
void init( po::options_description & desc ){
    desc.add_options()
        ("xfile,x", value<string>()->required(), "x data (row-wise nExamples x dim)")
        ("solnfile,s", value<string>()->required(), "solnfile[.soln] starting solver state")
        ("yfile,y", value<string>()->default_value(string("")), "optional y data (slc/mlc SparseMb)")
        ("threads,t", value<uint32_t>()->default_value(1U), "threads")
        ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "col-normalize x dimensions (mean=stdev=1)")
        ("help,h", value<bool>()->implicit_value(true), "this help")
        ;
}

void argsParse( int argc, char**argv ){
#define ARGSDEBUG 1
#if ARGSDEBUG > 0
    cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
    for( int i=0; i<argc; ++i ) {
        cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
    }
#endif
    bool keepgoing = true;
    try {
        po::options_description descMcproj("mcproj options");
        init( descMcproj );                        // create a description of the options

        po::variables_map vm;
        //po::store( po::parse_command_line(argc,argv,desc), vm );
        // Need more control, I think...
        {
            po::parsed_options parsed
                = po::command_line_parser( argc, argv )
                .options( descMcproj )
                //.positional( po::positional_options_description() ) // empty, none allowed.
                //.allow_unregistered()
                .run();
            po::store( parsed, vm );
        }

        if( vm.count("help") ) {
            helpUsage( cout );
            cout<<descMcproj<<endl;
            //helpExamples(cout);
            keepgoing=false;
        }

        po::notify(vm); // at this point, raise any exceptions for 'required' args

        cerr<<"mcproj args..."<<endl;
        assert( vm.count("xfile") );
        assert( vm.count("solnfile") );

        xFile=vm["xfile"].as<string>();
        solnFile=vm["solnfile"].as<string>();

        yFile=vm["yfile"].as<string>();
        xnorm=vm["xnorm"].as<bool>();
        threads=vm["threads"].as<uint32_t>();
        if( solnFile.rfind(".soln") != solnFile.size() - 5U ) solnFile.append(".soln");
    }catch(po::error& e){
        cerr<<"Invalid argument: "<<e.what()<<endl;
        throw;
    }catch(...){
        cerr<<"Command-line parsing exception of unknown type!"<<endl;
        throw;
    }
    if( ! keepgoing ) exit(0);
    return;
}

int main(int argc, char**argv){
    argsParse(argc,argv);
    MCsoln soln;
    {
        try{
            ifstream sfs(solnFile);
            if(!sfs.good()) {cerr<<" solnfile "<<solnFile<<endl; throw std::runtime_error("trouble opening soln file");}
            soln.read( sfs );
            sfs.close();
        }catch(std::exception const& e){
            cerr<<"Error reading solnfile "<<solnFile<<endl<<e.what()<<endl;
            throw;
        }
    }
    DenseM xDense;
    bool denseOk=false;
    SparseM xSparse;
    bool sparseOk=false;
    {
        ifstream xfs;
        // TODO XXX try Dense-Text, Sparse-Text too?
        try{
            xfs.open(xFile);
            if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
            // Note: after read rows,cols, COULD have alternate reader that checks for exactly correct file size
            ::detail::eigen_io_bin(xfs, xDense);
            if( xfs.fail() ) throw std::underflow_error("problem reading DenseM from xfile with eigen_io_bin");
            assert( xDense.cols() > 0U );
            char c;
            xfs >> c;   // should trigger eof if BINARY dense file exactly the write length
            if( ! xfs.eof() ) throw std::overflow_error("xDense read did not use full file");
            xfs.close();
            denseOk=true;
        }catch(po::error& e){
            cerr<<"Invalid argument: "<<e.what()<<endl;
            xfs.close();
            throw;
        }catch(std::exception const& e){
            cerr<<" xFile as DenseM failed: "<<e.what()<<endl;
            xDense.resize(0,0); //denseOk=false;
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
    SparseMb y;                         // now this is optional
    bool yOk=false;
    if(yFile.size()){
        ifstream yfs;
        try{
            yfs.open(yFile);
            if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
            ::detail::eigen_io_binbool( yfs, y );
            assert( y.cols() > 0U );
            if( yfs.fail() ) throw std::underflow_error("problem reading yfile with eigen_io_binbool");
            yOk=true;
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

    VectorXd xmean;
    VectorXd xstdev;
    VectorXsz no_active;

    if( denseOk ){
        if( xnorm ){
            cout<<"xDense ORIG:\n"<<xDense<<endl;
            cout<<" xnorm!"<<endl;
#if 1 // perhaps a better alg. but I only programmed it for dense, so far
            column_normalize(xDense,xmean,xstdev);
            cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
            cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
#else
            normalize_col(xDense);
#endif
        }
        cout<<"xDense"<<prettyDims(xDense)<<":\n"<<xDense<<endl;
        cout<<"y"<<prettyDims(y)<<":\n"<<y<<endl;
        assert( !yOk || xDense.rows() == y.rows() );
        if(1){ // see that one of the low-level routines runs...
            // Basic operation
            ActiveDataSet* ads = getactive( no_active, xDense,
                                            soln.weights_avg, soln.lower_bounds_avg, soln.upper_bounds_avg,
                                            /*verbose=*/true );
            assert( ads != nullptr );
            free_ActiveDataSet(ads); delete ads; ads=nullptr;
            // ? y should be fully optional
            //PredictionSet* pds = predict( xDense, y, ads, no_active, /*verbose=*/true /*...*/ );
            //
            //  evaluate_projections calculates much more...
            //    it seems to give a float weight to each assignment as well !
            //
            //evaluate_projections( xDense,y,soln ); // ???
            //
            cout<<"Good: got back from original 'getactive' call"<<endl;
        }
        if(1){ // new API: mcpredict.h
            vector<boost::dynamic_bitset<>> filt = project( xDense, soln );
            if(1) for(uint32_t i=0U; i<filt.size(); ++i){
                auto const& fi = filt[i];
                cout<<" x.row("<<setw(4)<<i<<")=";
                OUTWIDE(cout,40,xDense.row(i));
                cout<<"   classes: ";
                for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) cout<<" "<<c;
                cout<<endl;
            }
            if(0) for(uint32_t i=0U; i<filt.size(); ++i){
                auto const& fi = filt[i];
                cout<<setw(5)<<i<<" "<<fi<<endl; // ** reverse ** order of classes (msb-first)
            }
        }
                
    }else if( sparseOk ){
#if 1
#if 1 // same as dense ???
        vector<boost::dynamic_bitset<>> filt = project( xSparse, soln );
        if(1) for(uint32_t i=0U; i<filt.size(); ++i){
            auto const& fi = filt[i];
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
#endif
        cerr<<" sparse mcproj TBD"<<endl;
        exit(0);
#else
        assert( ! xnorm );
        //if( xnorm ){ cout<<" xnorm!"<<endl; column_normalize(xSparse,xmean,xstdev); } // col-norm DISALLOWED
        cout<<"xSparse:\n"<<xSparse<<endl;
        cout<<"y:\n"<<y<<endl;
        cout<<"parms:\n"<<parms<<endl;
        assert( xSparse.rows() == y.rows() );
        mcsolver.solve( xSparse, y, &parms );
#endif
    }
    if(1){ // output something about the soln
        DenseM      & w = soln.weights_avg;
        DenseM      & ww = soln.weights;
        w.colwise().normalize();                     // modify w, ww to unit-vector (? largest coeff +ve ?)
        ww.colwise().normalize();
        DenseM const& l = soln.lower_bounds_avg;
        DenseM const& u = soln.upper_bounds_avg;
        if(1){
            cout<<"     weights    "<<prettyDims(ww)<<":\n"<<ww<<endl;
            cout<<"     weights_avg"<<prettyDims(w)<<":\n"<<w<<endl;
            cout<<"lower_bounds_avg"<<prettyDims(l)<<":\n"<<l<<endl;
            cout<<"upper_bounds_avg"<<prettyDims(u)<<":\n"<<u<<endl;
        }
        if(1){
            for(int p=0U; p<w.cols(); ++p){   // for each projection
                cout<<" Projection "<<p<<" weights "<< w.col(1).transpose();
                uint32_t c=0U;
                for(uint32_t c=0U; c<l.rows(); ++c){ // for each class
                    if( c%8U == 0U ) {cout<<"\n {l,u} "<<setw(4)<<c;}
                    cout<<" { "<<setw(9)<<l.coeff(c,p)<<","<<setw(9)<<u.coeff(c,p)<<"}";
                }
                if(c%8U==0U) cout<<endl;
            }
        }
    }
    cout<<"\nGoodbye"<<endl;
}
