/** \file
 * <B>M</B>ulti-<B>C</B>lass <B>solver</B> for discriminatin projection lines.
 */

#include "find_w.hh"
#include "parameter-args.h"
#include "normalize.h"

#include <string>
#include <fstream>

using namespace std;
using namespace boost::program_options;
namespace po = boost::program_options;

param_struct parms;     ///< solver parameters, \ref parameter.h
string xFile;           ///< x data file name (io via ???)
string yFile;           ///< y data file name (io via eigen_io_binbool)
string solnFile;        ///< solution file basename
string outFile;         ///< solution file basename
bool xnorm=false;       ///< normalize x dims across examples to mean and stdev of 1.0

void helpUsage( std::ostream& os ){
    os  <<" Function:"
        <<"\n    Determine a number (--proj) of projection lines. Any example whose projection"
        <<"\n    lies outside bounds for that class is 'filtered'.  With many projecting lines,"
        <<"\n    potentially many classes can be filtered out, leaving few potential classes"
        <<"\n    for each example."
        <<"\n Usage:"
        <<"\n    mcsolve --xfile=... --yfile=... [--solnfile=...] [--output=...] [solver args...]"
        <<"\n where solver args guide the optimization procedure"
        <<"\n Without solnfile[.soln], use random initial conditions."
        <<"\n xfile is a plain eigen DenseM or SparseM (always stored as float)"
        <<"\n yfile is an Eigen SparseMb matrix of bool storing only the true values,"
        <<"\n       read/written via 'eigen_io_binbool'"
        <<endl;
}
void init( po::options_description & desc ){
    desc.add_options()
        ("xfile,x", value<string>()->required(), "x data (row-wise nExamples x dim)")
        ("yfile,y", value<string>()->required(), "y data (slc/mlc SparseMb)")
        ("solnfile,s", value<string>()->default_value(string("")), "solnfile[.soln] starting solver state")
        ("output,o", value<string>()->default_value(string("mc")), "output[.soln] file base name")
        ("threads,t", value<uint32_t>()->default_value(1U), "threads")
        ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "col-normalize x dimensions (mean=stdev=1)")
        ;
}

#if 0 // moved to normalization.h (and into library as column_normalize(Dense const& x, VectorXd& mean, VectorXd& stdev))
template< typename DERIVED >
void column_mean_stdev( Eigen::DenseBase<DERIVED> const& x, VectorXd & mean, VectorXd &var ){
    mean.resize( x.cols() );
    mean.setZero();
    var.resize( x.cols() );
    var.setZero();
    size_t n=0U;
    VectorXd delta( x.cols() );
    for(uint32_t r=0U; r<x.rows(); ++r){
        ++n;
        delta = x.row(r).transpose() - mean;
        mean += delta * (1.0/n);
        var += delta.cwiseProduct( x.row(r).transpose() - mean );
    }
    if( n < 2 ) var.setOnes();                  // not enough data to calculate var
    else var = (var * (1.0/(n-1))).cwiseSqrt(); // var --> correct scale --> stdev
}
#endif

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
        po::options_description descAll("Allowed options");
        po::options_description descMcsolve("mcsolve options");
        init( descMcsolve );                        // create a description of the options
        po::options_description descParms("solver args");
        opt::mcParameterDesc( descParms );          // add the param_struct options
        descAll.add(descMcsolve).add(descParms);

        po::variables_map vm;
        //po::store( po::parse_command_line(argc,argv,desc), vm );
        // Need more control, I think...
        {
            po::parsed_options parsed
                = po::command_line_parser( argc, argv )
                .options( descAll )
                //.positional( po::positional_options_description() ) // empty, none allowed.
                //.allow_unregistered()
                .run();
            po::store( parsed, vm );
        }

        if( vm.count("help") ) {
            helpUsage( cout );
            cout<<descAll<<endl;
            //helpExamples(cout);
            keepgoing=false;
        }

        po::notify(vm); // at this point, raise any exceptions for 'required' args

        cerr<<"msolve args..."<<endl;
        assert( vm.count("xfile") );
        assert( vm.count("yfile") );
        assert( vm.count("solnfile") );
        assert( vm.count("output") );

        xFile=vm["xfile"].as<string>();
        yFile=vm["yfile"].as<string>();
        solnFile=vm["solnfile"].as<string>();
        outFile=vm["output"].as<string>();
        xnorm=vm["xnorm"].as<bool>();

        if( solnFile.rfind(".soln") != solnFile.size() - 5U ) solnFile.append(".soln");
        if( outFile .rfind(".soln") != outFile .size() - 5U ) outFile .append(".soln");

        cerr<<"opt::extract..."<<endl;
        opt::extract(vm,parms);         // retrieve McSolver parameters
    }
    catch(po::error& e)
    {
        cerr<<"Invalid argument: "<<e.what()<<endl;
        throw;
    }
#if 1
    catch(...)
    {
        cerr<<"Command-line parsing exception of unknown type!"<<endl;
        throw;
    }
#endif
    if( ! keepgoing ) exit(0);
    return;
}

int main(int argc, char**argv){
    argsParse(argc,argv);
    MCsolver mcsolver( solnFile.size()? solnFile.c_str(): (char const* const)nullptr);
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
            ::detail::eigen_io_bin(xfs, xDense);
            xfs.close();
            if( xfs.fail() ) throw std::underflow_error("problem reading DenseM from xfile with eigen_io_bin");
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
    SparseMb y;
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

    VectorXd xmean;
    VectorXd xstdev;
    if( denseOk ){
        if( xnorm ){
            cout<<"xDense ORIG:\n"<<xDense<<endl;
            cout<<" xnorm!"<<endl;
#if 1
            column_normalize(xDense,xmean,xstdev);
            cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
            cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
#else
            normalize_col(xDense);
#endif
        }
        cout<<"xDense:\n"<<xDense<<endl;
        cout<<"y:\n"<<y<<endl;
        cout<<"parms:\n"<<parms<<endl;
        assert( xDense.rows() == y.rows() );
        mcsolver.solve( xDense, y, &parms );
    }else if( sparseOk ){
        assert( ! xnorm );
        //if( xnorm ){ cout<<" xnorm!"<<endl; column_normalize(xSparse,xmean,xstdev); } // col-norm DISALLOWED
        cout<<"xSparse:\n"<<xSparse<<endl;
        cout<<"y:\n"<<y<<endl;
        cout<<"parms:\n"<<parms<<endl;
        assert( xSparse.rows() == y.rows() );
        mcsolver.solve( xSparse, y, &parms );
    }
    if(1){ // output something about the soln
        MCsoln & soln = mcsolver.getSoln();
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
