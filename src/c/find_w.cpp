
#include "find_w.hh"            // get template definition of the solve_optimization problem
#include "mcsolver.hh"          // template impl of MCsolver version
#include "normalize.h"          // MCxyData support funcs
#include "constants.h" // MCTHREADS
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

// Explicitly instantiate templates into the library

// ----------------- Eigen native Dense and Sparse --------------
template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        const DenseM& x,                        // Dense
                        const SparseMb& y,
                        const param_struct& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        const SparseM& x,                       // Sparse
                        const SparseMb& y,
                        const param_struct& params);

// ---------------- External Memory variants --------------------
template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        //ExtConstDenseM& x,                    // type lookup --> no match
                        Eigen::Map<DenseM const> const& x,      // external-memory Dense
                        SparseMb const& y,
                        param_struct const& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        Eigen::MappedSparseMatrix<double, Eigen::RowMajor> const& x,// double const WON'T WORK
                        SparseMb const& y,
                        param_struct const& params);
                        

// Complete some of the class declarations before instantiating MCsolver
MCsolver::MCsolver(char const* const solnfile /*= nullptr*/)
: MCsoln()
    // private "solve" variables here TODO
{
    if( solnfile ){
        ifstream ifs(solnfile);
        if( ifs.good() ) try{
            cout<<" reading "<<solnfile<<endl;
            this->read( ifs );
            this->pretty( cout );
            cout<<" reading "<<solnfile<<" DONE"<<endl;
        }catch(std::exception const& e){
            ostringstream err;
            err<<"ERROR: unrecoverable error reading MCsoln from file "<<solnfile;
            throw(runtime_error(err.str()));
        }
    }
}
MCsolver::~MCsolver()
{
    //cout<<" ~MCsolver--TODO: where to write the MCsoln ?"<<endl;
}


// Explicitly instantiate MCsolver into the library

template
void MCsolver::solve( DenseM const& x, SparseMb const& y, param_struct const* const params_arg );
template
void MCsolver::solve( SparseM const& x, SparseMb const& y, param_struct const* const params_arg );
template
void MCsolver::solve( ExtConstSparseM const& x, SparseMb const& y, param_struct const* const params_arg );

void MCsolver::trim( enum Trim const kp ){
    if( kp == TRIM_LAST ){
        // If have some 'last' data, swap {w,l,u} into {w,l,u}_avg
        if( weights.size() != 0 ){
            weights_avg.swap(weights);
            lower_bounds_avg.swap(lower_bounds);
            upper_bounds_avg.swap(upper_bounds);
        }
    }
    // ** ALL ** the non-SHORT MCsoln memory is freed
    // NOTE: in Eigen. resize always reallocates memory, so resize(0) WILL free memory.
    objective_val_avg.resize(0);
    weights.resize(0,0);
    lower_bounds.resize(0,0);
    upper_bounds.resize(0,0);
    objective_val.resize(0);
}


// -------- MCsoln I/O --------
using namespace std;
using namespace detail;     // see printing.h

std::array<char,4> MCsoln::magicTxt = {'M', 'C', 's', 't' };
std::array<char,4> MCsoln::magicBin = {'M', 'C', 's', 'b' };
std::array<char,4> MCsoln::magicCnt = {'M', 'C', 's', 'c' };
std::array<char,4> MCsoln::magicEof = {'M', 'C', 's', 'z' };
//#define MAGIC_U32( ARRAY4C ) (*reinterpret_cast<uint_least32_t const*>( ARRAY4C.cbegin() ))
#define MAGIC_EQU( A, B ) (A[0]==B[0] && A[1]==B[1] && A[2]==B[2] && A[3]==B[3])

MCsoln::MCsoln()
: magicHdr( magicBin )
    , d( 0U )           // unknown until first call to 'solve', when have training examples 'x'
    , nProj( 0U )
    , nClass( 0U )
    , fname("")

    , parms(set_default_params())

    , t( 0U )
    , C1( parms.C1 )
    , C2( parms.C2 )    // actually this is never modified (error?)
    , lambda( 1.0 / parms.C2 )
    , eta_t( parms.eta )
    , magicData( magicCnt )

    , weights_avg()
    , lower_bounds_avg()
    , upper_bounds_avg()
    , magicEof1( magicCnt )     // default SHORT i/o would stop here

    , objective_val_avg()
    , magicEof2( magicCnt )

    , weights()
    , lower_bounds()
    , upper_bounds()
    , objective_val()
    , magicEof3( magicEof )     // default SHORT i/o would stop here
{};
void MCsoln::write( std::ostream& os, enum Fmt fmt/*=BINARY*/, enum Len len/*=SHORT*/ ) const
{
    if(fmt!=BINARY){
        io_txt(os,magicTxt);
        write_ascii(os,len);
    }else{
        io_bin(os,magicBin);
        write_binary(os,len);
    }
}

void MCsoln::read( std::istream& is ){
    io_bin(is,magicHdr);
    if(      MAGIC_EQU(magicHdr,magicTxt) ) read_ascii( is );
    else if( MAGIC_EQU(magicHdr,magicBin) ) read_binary( is );
    else throw std::runtime_error("ERROR: bad magicHdr reading MCsoln");
}
void MCsoln::pretty( std::ostream& os ) const {
    os<<"--------- d="<<d<<" nProj="<<nProj<<" nClass="<<nClass<<" fname='"<<fname<<"'"<<endl;
    os<<"--------- MCsolver parameters:\n"<<parms; //<<endl;
    os<<"--------- last iter t="<<t<<" C1="<<C1<<" C2="<<C2<<" lambda="<<lambda<<" eta_t="<<eta_t<<endl;
    os<<"--------- weights_avg"<<prettyDims(weights_avg)
        <<" lower_bounds_avg"<<prettyDims(lower_bounds_avg)
        <<" upper_bounds"<<prettyDims(upper_bounds_avg)<<endl;
    os<<"--------- weights_avg:\n"<<weights_avg<<endl;
    os<<"--------- lower_bounds_avg:\n"<<lower_bounds_avg<<endl;
    os<<"--------- upper_bounds_avg:\n"<<upper_bounds_avg<<endl;
    //os<<"--------- medians:\n"<<medians<<endl; // NOT YET THERE
}

struct MCLazyData {
    struct MeanStdev{
        size_t n;               ///< common vector size
        /** \f$\frac{1}{n}\sum_0^n x_i\f$. Valid for any \c n. */
        VectorXd mean;
        /** unbiased version, \f$\sigma^2=\frac{1}{n-1}\sum_1^n(x_i-\mu)^2\f$.
         * Meaningful only for n>1. */
        VectorXd stdev;
    };
    MCLazyData() : rowOk(false), colOk(false), row2Ok(false), col2Ok(false)
                   , row_mean0(false), row_stdev1(false), col_mean0(false), col_stdev1(false)
                   , row(nullptr), col(nullptr), row2(), col2() {}
    ~MCLazyData() { if(row){ delete row; row=nullptr; } if(col){ delete col; col=nullptr; } }
    // quick invalidation (release memory only when nec. hopefully in destructor)
    bool rowOk, colOk, row2Ok, col2Ok;
    // remember if any normalizations have been done
    bool row_mean0, row_stdev1, col_mean0, col_stdev1;
    struct MeanStdev *row;
    struct MeanStdev *col;
    /** \f$\lVert x \rVert_2^2 = \sum_1^n x_i^2\f$.
     * If we have \c mean and \c stdev already, we could
     * use \f$\sigma^2 = \frac{1}{n-1}\left((\sum_1^n x_i^2) - n\mu^2\right)\f$,
     * to set \c sumsqr with \f$\sum_1^n x_i^2 = (n-1)\sigma^2 + n(n-1)\mu^2\f$.
     * However, sumsqr is always valid (for any \c n).
     * (Using norm2 to help set MeanStdev is likely numerically less stable
     *  than the normal MeanStdev method)
     */
    VectorXd row2; ///< row l2-norms
    VectorXd col2; ///< column l2-norms

    /** invalidate all stats (doesn't free memory) */
    void changed(){
        rowOk = colOk = row2Ok = col2Ok = row_mean0 = row_stdev1 = col_mean0 = col_stdev1 = false;
    }
    MeanStdev& freshRow( size_t const n ){
        if( !row ) row = new MeanStdev();
        row->n = n;
        row->mean.resize(0); row->mean.resize(n);
        row->stdev.resize(0); row->stdev.resize(n);
        return *row;
    }
    MeanStdev& freshCol( size_t const n ){
        if( !col ) col = new MeanStdev();
        col->n = n;
        col->mean.resize(0); col->mean.resize(n);
        col->stdev.resize(0); col->stdev.resize(n);
        return *col;
    }
    void setRow2( DenseM const &x );
    void setRow2( SparseM const &x );
    void setCol2( DenseM const &x );
    //void setRow2( SparseM const &x );
    void runit( DenseM & x );   // make x rows into unit vectors by scaling
    void runit( SparseM & x );
    void cunit( DenseM & x );   // make x cols into unit vectors by scaling
    void cunit( SparseM & x );  ///< Just throws a runtime_error.
    void setMeanStdev( DenseM const& x ) {cout<<"setRow2 TBD"<<endl;}
    void setMeanStdev( SparseM const& x ) {cout<<"setRow2 TBD"<<endl;}
};
MCxyData::MCxyData() : xDense(), denseOk(false), xSparse(), sparseOk(false)
                       , y(), qscal(0.0), xscal(0.0)
                       , lazx(nullptr), lazy(nullptr) {}
MCxyData::~MCxyData(){
    delete lazx; lazx=nullptr;
    delete lazy; lazy=nullptr;
}
void MCxyData::xchanged() {
    if( lazx ) { lazx->changed();}
}
void MCxyData::ychanged() {
    if( lazy ) { lazy->changed(); }
}
void MCxyData::xscale( double const mul ){
    if(mul != 1.0){
        if( sparseOk ) xSparse *= mul;
        else           xDense  *= mul;
        if(xscal==0.0) xscal=1.0;
        xscal *= mul;
        qscal *= mul;
        xchanged();
    }
}
void MCLazyData::setRow2( DenseM const &x ){
    row2.resize(0); row2.resize(x.rows());
    row2 = x.rowwise().norm(); // TODO optimize if rowOk (faster)
    row2Ok = true; col2Ok = false;
}
void MCLazyData::setRow2( SparseM const &x ){
    row2.resize(0); row2.resize(x.rows());
    // No: row2 = x.rowwise().norm();
    // optimize if rowOk!
    for(SparseM::Index i=0; i<x.rows(); ++i){
        row2.coeffRef(i) = x.row(i).norm();
    }
    row2Ok = true; col2Ok = false;
}
void MCLazyData::setCol2( DenseM const &x ){
    col2.resize(0); col2.resize(x.rows());
    col2 = x.colwise().norm(); // TODO optimize if rowOk (faster)
    col2Ok = true; row2Ok = false;
}
void MCLazyData::runit( DenseM& x ){
    setRow2( x );
    for(size_t r=0U; r<x.rows(); ++r){
        double const f = 1.0 / row2.coeff(r);
        if( !(f < 1.e-10) ){
            x.row(r) *= f;
            row2.coeffRef(r) = 1.0;
        }
    }
}
void MCLazyData::runit( SparseM& x ){
    setRow2( x );
    for(size_t r=0U; r<x.rows(); ++r){
        double const f = 1.0 / row2.coeff(r);
        if( !(f < 1.e-10) ){
            x.row(r) *= f;
            row2.coeffRef(r) = 1.0;
        }
    }
}
void MCLazyData::cunit( DenseM& x ){
    setCol2( x );
    for(size_t r=0U; r<x.rows(); ++r){
        double const f = 1.0 / col2.coeff(r);
        if( !(f < 1.e-10) ){
            x.col(r) *= f;
            col2.coeffRef(r) = 1.0;
        }
    }
}
void MCLazyData::cunit( SparseM& x ){
    throw std::runtime_error("col unit of SparseM not supported");
}


void MCxyData::xrunit(){
    if(!lazx) lazx = new MCLazyData();
    if( denseOk ) lazx->runit(xDense);
    else          lazx->runit(xSparse);
}

void MCxyData::xcunit(){
    if(!lazx) lazx = new MCLazyData();
    if( denseOk ) lazx->cunit(xDense);
    else          lazx->cunit(xSparse);
}

void MCxyData::xrnormal(){
    if( sparseOk ) throw std::runtime_error("row norm of SparseM? convert to DenseM, please");
    if( !denseOk ) throw std::runtime_error("no x data for xrnormal");
    if( lazx && lazx->row_mean0 && lazx->row_stdev1 )
        throw std::runtime_error("possible double call to xrnormal?");
    xchanged();
    if(!lazx) lazx = new MCLazyData();
    auto& ms = lazx->freshRow( xDense.rows() );    // a MeanStdev
    row_normalize( xDense, ms.mean, ms.stdev );  // save OLD mean stdev
    lazx->rowOk = lazx->row_mean0 = lazx->row_stdev1 = true;
    lazx->colOk = lazx->col_mean0 = lazx->col_stdev1 = false;
    lazx->row2.resize(0); lazx->row2.resize(xDense.rows());
    lazx->row2.setOnes();
    lazx->row2Ok = true;
    lazx->col2Ok = false;
}
void MCxyData::xcnormal(){
    if( sparseOk ) throw std::runtime_error("x col norm of SparseM? convert to DenseM, please");
    if( !denseOk ) throw std::runtime_error("no x data for xrnormal");
    if( lazx && lazx->col_mean0 && lazx->col_stdev1 )
        throw std::runtime_error("possible double call to xcnormal?");
    xchanged();
    if(!lazx) lazx = new MCLazyData();
    auto& ms = lazx->freshCol( xDense.cols() );    // a MeanStdev
    col_normalize( xDense, ms.mean, ms.stdev );      // save OLD mean stdev
    lazx->colOk = lazx->col_mean0 = lazx->col_stdev1 = true;
    lazx->rowOk = lazx->row_mean0 = lazx->row_stdev1 = false;
    lazx->col2.resize(0); lazx->col2.resize(xDense.cols());
    lazx->col2.setOnes();
    lazx->col2Ok = true;
    lazx->row2Ok = false;
}
void MCxyData::xread( std::string xFile ){
    ifstream xfs;
    std::array<char,4> magicHdr;
    // TODO XXX try Dense-Text, Sparse-Text too?
    try{
        xfs.open(xFile);
        io_bin(xfs,magicHdr);
        if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile");
        if( MAGIC_EQU(magicHdr,MCxyData::magic_xDense)){
            ::detail::eigen_io_bin(xfs, xDense);
            if( xfs.fail() ) throw std::underflow_error("problem reading DenseM from xfile with eigen_io_bin");
            char c;
            xfs >> c;   // should trigger eof if BINARY dense file exactly the write length
            if( ! xfs.eof() ) throw std::overflow_error("xDense read did not use full file");
            xfs.close();
            assert( xDense.cols() > 0U );
            denseOk=true;
        }else if( MAGIC_EQU(magicHdr,MCxyData::magic_xSparse)){
            ::detail::eigen_io_bin( xfs, xSparse );
            if( xfs.fail() ) throw std::underflow_error("problem reading SparseM from xfile with eigen_io_bin");
            xfs.close();
            assert( xSparse.cols() > 0U );
            sparseOk=true;
        }else{
            cerr<<" Neither sparse nor dense binary x data was detected"<<endl;
            // Not here [yet]: text formats? libsvm? milde repo?
        }
    }catch(std::exception const& e){
        cout<<" oops reading x data from "<<xFile<<" ... "<<e.what()<<endl;
        throw;
    }
}
void MCxyData::xwrite( std::string fname ) const { // write binary (either sparse/dense)
    ofstream ofs;
    try{
        if( !sparseOk && !denseOk )
            throw std::runtime_error("savex no Eigen x data yet");
        ofs.open(fname);
        if( ! ofs.good() ) throw std::runtime_error("savex trouble opening fname");
        if( sparseOk ){
            io_bin(ofs,MCxyData::magic_xSparse);
            detail::eigen_io_bin(ofs, xSparse);
        }else{ assert(denseOk);
            io_bin(ofs,MCxyData::magic_xSparse);
            detail::eigen_io_bin(ofs, xDense);
        }
        if( ! ofs.good() ) throw std::runtime_error("savex trouble writing fname");
        ofs.close();
    }catch(std::exception const& e){
        cerr<<" trouble writing "<<fname<<" : unknown exception"<<endl;
        ofs.close();
        throw;
    }
}
void MCxyData::yread( std::string yFile ){
    std::array<char,4> magicHdr;
    ifstream yfs;
    bool yOk = false;
    try{
        yfs.open(yFile);
        if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
        io_bin(yfs,magicHdr);
        if( MAGIC_EQU(magicHdr,magic_yBin) ){
            ::detail::eigen_io_binbool( yfs, y );
            assert( y.cols() > 0U );
            if( yfs.fail() ) throw std::underflow_error("problem reading yfile with eigen_io_binbool");
            yOk = true;
        }
    }catch(std::runtime_error const& e){
        cerr<<e.what()<<endl;
        //throw; // continue execution -- try text format
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
            yOk=true;
        }
        catch(std::exception const& e){
            cerr<<" --file could not be read in text mode from "<<yFile<<" -- "<<e.what()<<endl;
            throw;
        }
    }
}
void MCxyData::ywrite( std::string yFile ) const { // write binary y data (not very compact!)
    ofstream ofs;
    try{
        ofs.open(yFile);
        if( ! ofs.good() ) throw std::runtime_error("ywrite trouble opening file");
        io_bin(ofs,MCxyData::magic_yBin);
        detail::eigen_io_binbool(ofs, y);
        if( ! ofs.good() ) throw std::runtime_error("ywrite trouble writing file");
        ofs.close();
    }catch(std::exception const& e){
        cerr<<" ywrite trouble writing "<<yFile<<" : unknown exception"<<e.what()<<endl;
        ofs.close();
        throw;
    }
}
std::string MCxyData::shortMsg() const {
    ostringstream oss;
    if( denseOk ) oss<<" dense x "<<prettyDims(xDense);
    if( sparseOk ) oss<<" sparse x "<<prettyDims(xSparse);
    oss<<" SparseMb y "<<prettyDims(y);
    return oss.str();
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
/** This can be very slow and should always be
 * done with full parallelism.
 * \throw if no data. */
void MCxyData::quadx(double qscal /*=0.0*/){
    if( !sparseOk && !denseOk ) throw std::runtime_error(" quadx needs x data");
    if( this->qscal != 0.0 ) throw std::runtime_error("quadx called twice!?");
    // divide xi * xj terms by max|x| ?
    // look at quad terms globally and make them mean 0 stdev 1 ?
    // scale so mean is 0 and all values lie within [-2,2] ?
    if( qscal==0.0 ){
        // calculate some "max"-ish value, and invert it.
        // I'll use l_{\inf} norm, or max absolute value,
        // so as not to ruin sparsity (if any)
        if( sparseOk ){
            if(!xSparse.isCompressed()) xSparse.makeCompressed();
#if 0 // not available for sparse..   l_{\inf} norm == max absolute value
            qscal = xSparse.lpNorm<Eigen::Infinity>();
#else
            SparseM::Index const nData = xSparse.outerIndexPtr()[ xSparse.outerSize() ];
            auto       b = xSparse.valuePtr();
            auto const e = b + nData;
            SparseM::Scalar maxabs=0.0;
            for( ; b<e; ++b) maxabs = std::max( maxabs, std::abs(*b) );
            qscal = maxabs;
#endif
        }else{
            // not supported: qscal = xDense.cwiseAbs().max();
            qscal = xDense.lpNorm<Eigen::Infinity>();
        }
        qscal = 1.0/qscal;
        // (note all compares with NaN are false, except !=)
        if( !(qscal > 1.e-12) ) qscal=1.0; // ignore strange/NaN scaling
    }
    xchanged(); // remove any old x stats
    if( sparseOk ) addQuadratic( xSparse, qscal );
    else           addQuadratic( xDense,  qscal );
    this->qscal = qscal;
}

// private post-magicHdr I/O routines ...
#define CHK_MAT_DIM(MAT,R,C,ERRMSG) do { \
    if( MAT.rows() != (R) || MAT.cols() != (C) ){ \
        std::ostringstream oss; \
        oss<<ERRMSG<<"\n\tmatrix["<<MAT.rows()<<"x"<<MAT.cols()<<"] expected ["<<(R)<<"x"<<(C)<<"]\n"; \
        throw std::runtime_error(oss.str()); \
    }}while(0)
#define CHK_VEC_DIM(VEC,SZ,ERRMSG) do { \
    if( VEC.size() != (SZ) ) { \
        std::ostringstream oss; \
        oss<<ERRMSG<<"\n\tvector["<<VEC.size()<<"] expected ["<<(SZ)<<"]\n"; \
        throw std::runtime_error(oss.str()); \
    }}while(0)
#define RETHROW( ERRMSG ) std::cerr<<e.what(); throw std::runtime_error(ERRMSG)

void MCsoln::write_ascii( std::ostream& os, enum Len len/*=SHORT*/ ) const
{
    try{
        io_txt(os,d);
        io_txt(os,nProj);
        io_txt(os,nClass);
        io_txt(os,fname);
        ::write_ascii(os,parms);
        io_txt(os,t);
        io_txt(os,C1);
        io_txt(os,C2);
        io_txt(os,lambda);
        io_txt(os,eta_t);
        magicData = magicCnt;
        io_txt(os,magicData);
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data write error");
    }

#define IO_MAT( OS, MAT, R, C, ERRMSG ) do{eigen_io_txt(OS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
#define IO_VEC( OS, VEC, SZ, ERRMSG ) do {eigen_io_txt(OS,VEC); CHK_VEC_DIM( VEC,SZ,ERRMSG );}while(0)
    try{
        IO_MAT(os, weights_avg     , d     , nProj, "write_ascii Bad weights_avg dimensions");
        IO_MAT(os, lower_bounds_avg, nClass, nProj, "write_ascii Bad lower_bounds_avg dimensions");
        IO_MAT(os, upper_bounds_avg, nClass, nProj, "write_ascii Bad upper_bounds_avg dimensions");
        magicEof1 = magicEof;
        io_txt(os,magicEof1);
    } catch(exception const& e){ RETHROW("MCsoln SHORT data read error"); }
    if( len == SHORT )
        return;

    try{
        IO_VEC(os, objective_val_avg, nClass, "Bad objective_val_avg dimension");
        magicEof2 = magicCnt;
        io_txt(os,magicEof2);
    }catch(exception const& e){ RETHROW("MCsoln optional data write error"); }

    try{
        IO_MAT(os, weights      , d     , nProj, "write_ascii Bad weights dimensions");
        IO_MAT(os, lower_bounds , nClass, nProj, "write_ascii Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds , nClass, nProj, "write_ascii Bad lower_bounds dimensions");
        IO_VEC(os, objective_val, nClass, "Bad objective_val dimensions");
        magicEof3 = magicEof;
        io_txt(os, magicEof3);
    }catch(exception const& e){ RETHROW("MCsoln bad long data write"); }
#undef IO_VEC
#undef IO_MAT
}
void MCsoln::read_ascii( std::istream& is ){
    try{
        io_txt(is,d);
        io_txt(is,nProj);
        io_txt(is,nClass);
        io_txt(is,fname);
        ::read_ascii(is,parms);
        io_txt(is,t);
        io_txt(is,C1);
        io_txt(is,C2);
        io_txt(is,lambda);
        io_txt(is,eta_t);
        //is>>std::skipws;                                      // <--- did not skip the LF ???
        while(is.peek()=='\n'|| is.peek()==' ') is.get();
        //cout<<"read_ascii magicData"<<endl;
        io_txt(is,magicData);
        //cout<<" read magicData as <"<<magicData[0]<<magicData[1]<<magicData[2]<<magicData[3]<<">"<<endl;
        //cout<<"      magicCnt  is <"<<magicCnt [0]<<magicCnt [1]<<magicCnt [2]<<magicCnt [3]<<">"<<endl;
        if( ! MAGIC_EQU(magicData,magicCnt) )
            throw runtime_error("MCsoln read_ascii header length has changed");
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data read error");
    }
    int keep_going, eof;
#define IO_MAT( IS, MAT, R, C, ERRMSG ) do{eigen_io_txt(IS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
#define IO_VEC( IS, VEC, SZ, ERRMSG ) do {eigen_io_txt(IS,VEC); CHK_VEC_DIM( VEC,SZ,ERRMSG );}while(0)
    try{
        //cout<<" pre-weights is.peek = 0x"<<hex<<uint32_t{is.peek()}<<" = <"<<is.peek()<<"> "<<endl;
        IO_MAT(is, weights_avg     , d     , nProj, "read_ascii Bad weights_avg dimensions");
        IO_MAT(is, lower_bounds_avg, nClass, nProj, "read_ascii Bad lower_bounds_avg dimensions");
        IO_MAT(is, upper_bounds_avg, nClass, nProj, "read_ascii Bad upper_bounds_avg dimensions");
        while(is.peek()=='\n'|| is.peek()==' ') is.get();       // <--- gotcha
        io_txt(is,magicEof1);
        eof        = is.eof() || MAGIC_EQU(magicEof1,magicEof);
        keep_going = ! eof && MAGIC_EQU(magicEof1,magicCnt);
        if( ! eof && ! keep_going ) throw runtime_error("MCsoln bad short data terminator");
    } catch(exception const& e){ RETHROW("MCsoln SHORT data read error"); }
    objective_val_avg.resize(0);
    weights          .resize(0,0);
    lower_bounds     .resize(0,0);
    upper_bounds     .resize(0,0);
    objective_val    .resize(0);
    if( !keep_going )
        return;
    // many long read errors are recoverable --- just resize as SHORT data.

    try{
        IO_VEC(is, objective_val_avg, nClass, "Bad objective_val_avg dimension");
        while(is.peek()=='\n'|| is.peek()==' ') is.get();       // <--- gotcha
        io_txt(is,magicEof2);
        eof        = is.eof() || MAGIC_EQU(magicEof2,magicEof);
        keep_going = ! eof && MAGIC_EQU(magicEof2,magicCnt);
        if( ! eof && ! keep_going ) throw runtime_error("MCsoln bad objective_val_avg terminator");
    }catch(exception const& e){
        cout<<e.what()<<"\n\tRecovering: ignoring optional long-format data";
        objective_val_avg.resize(0);
        keep_going = 0;
    }
    if( !keep_going )
        return;

    try{
        IO_MAT(is, weights     , d     , nProj, "read_ascii Bad weights dimensions");
        IO_MAT(is, lower_bounds, nClass, nProj, "read_ascii Bad lower_bounds dimensions");
        IO_MAT(is, upper_bounds, nClass, nProj, "read_ascii Bad lower_bounds dimensions");
        IO_VEC(is, objective_val, nClass, "read_ascii Bad objective_val dimensions");
        while(is.peek()=='\n'|| is.peek()==' ') is.get();       // <--- gotcha
        io_txt(is, magicEof3);
        eof        = is.eof() || MAGIC_EQU(magicEof1,magicEof) != 0;
        if( ! eof ) throw runtime_error("MCsoln bad LONG data terminator");
    }catch(exception const& e){
        cout<<e.what()<<"\n\tRecovering: ignoring optional long-format data";
        // many long read errors are recoverable --- just resize as SHORT data.
        weights      .resize(0,0);
        lower_bounds .resize(0,0);
        upper_bounds .resize(0,0);
        objective_val.resize(0);
        keep_going = 0;
    }
    return;
#undef IO_VEC
#undef IO_MAT
}
void MCsoln::write_binary( std::ostream& os, enum Len len/*=SHORT*/ ) const
{
    try{
        io_bin(os,d);
        io_bin(os,nProj);
        io_bin(os,nClass);
        io_bin(os,fname);
        ::write_binary(os,parms);
        io_bin(os,t);
        io_bin(os,C1);
        io_bin(os,C2);
        io_bin(os,lambda);
        io_bin(os,eta_t);
        magicData = magicCnt;
        io_bin(os,magicData);
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data write error");
    }

#define IO_MAT( OS, MAT, R, C, ERRMSG ) do{eigen_io_bin(OS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
#define IO_VEC( OS, VEC, SZ, ERRMSG ) do {eigen_io_bin(OS,VEC); CHK_VEC_DIM( VEC,SZ,ERRMSG );}while(0)
    try{
        IO_MAT(os, weights_avg     , d     , nProj, "write_binary Bad weights_avg dimensions");
        IO_MAT(os, lower_bounds_avg, nClass, nProj, "write_binary Bad lower_bounds_avg dimensions");
        IO_MAT(os, upper_bounds_avg, nClass, nProj, "write_binary Bad upper_bounds_avg dimensions");
        magicEof1 = magicEof;
        io_bin(os,magicEof1);
    } catch(exception const& e){ RETHROW("MCsoln SHORT data read error"); }
    if( len == SHORT )
        return;

    try{
        IO_VEC(os, objective_val_avg, nClass, "Bad objective_val_avg dimension");
        magicEof2 = magicCnt;
        io_bin(os,magicEof2);
    }catch(exception const& e){ RETHROW("MCsoln optional data write error"); }

    try{
        IO_MAT(os, weights     , d     , nProj, "write_binary Bad weights dimensions");
        IO_MAT(os, lower_bounds, nClass, nProj, "write_binary Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds, nClass, nProj, "write_binary Bad lower_bounds dimensions");
        IO_VEC(os, objective_val, nClass, "write_binary Bad objective_val dimensions");
        magicEof3 = magicEof;
        io_bin(os, magicEof3);
    }catch(exception const& e){ RETHROW("MCsoln bad long data write"); }
#undef IO_VEC
#undef IO_MAT
}
void MCsoln::read_binary( std::istream& is ){
    try{
        io_bin(is,d);
        io_bin(is,nProj);
        io_bin(is,nClass);
        io_bin(is,fname);
        ::read_binary(is,parms);
        io_bin(is,t);
        io_bin(is,C1);
        io_bin(is,C2);
        io_bin(is,lambda);
        io_bin(is,eta_t);
        io_bin(is,magicData);
        if( ! MAGIC_EQU(magicData,magicCnt) )
            throw runtime_error("MCsoln header length has changed");
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data read error");
    }
    int keep_going, eof;
#define IO_MAT( IS, MAT, R, C, ERRMSG ) do{eigen_io_bin(IS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
#define IO_VEC( IS, VEC, SZ, ERRMSG ) do {eigen_io_bin(IS,VEC); CHK_VEC_DIM( VEC,SZ,ERRMSG );}while(0)
    try{
        IO_MAT(is,      weights_avg, d     , nProj, "read_binary Bad weights_avg dimensions");
        IO_MAT(is, lower_bounds_avg, nClass, nProj, "read_binary Bad weights_avg dimensions");
        IO_MAT(is, upper_bounds_avg, nClass, nProj, "read_binary Bad weights_avg dimensions");
        io_bin(is,magicEof1);
        eof        = is.eof() || MAGIC_EQU(magicEof1,magicEof);
        keep_going = ! eof && MAGIC_EQU(magicEof1,magicCnt);
        if( ! eof && ! keep_going ) throw runtime_error("MCsoln bad short data terminator");
    } catch(exception const& e){ RETHROW("MCsoln SHORT data read error"); }
    //cout<<"read_binary short read ok, resizing things to 0 ??? "<<endl;
    objective_val_avg.resize(0);
    weights          .resize(0,0);
    lower_bounds     .resize(0,0);
    upper_bounds     .resize(0,0);
    objective_val    .resize(0);
    if( !keep_going )
        return;
    // many long read errors are recoverable --- just resize as SHORT data.

    try{
        IO_VEC(is, objective_val_avg, nClass, "Bad objective_val_avg dimension");
        io_bin(is,magicEof2);
        eof        = is.eof() || MAGIC_EQU(magicEof1,magicEof);
        keep_going = ! eof && MAGIC_EQU(magicEof1,magicCnt);
        if( ! eof && ! keep_going ) throw runtime_error("MCsoln bad objective_val_avg terminator");
    }catch(exception const& e){
        cout<<e.what()<<"\n\tRecovering: ignoring optional long-format data";
        objective_val_avg.resize(0);
        keep_going = 0;
    }
    if( !keep_going )
        return;

    try{
        IO_MAT(is,      weights , d     , nProj, "Bad weights dimensions");
        IO_MAT(is, lower_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_MAT(is, upper_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_VEC(is, objective_val, nClass, "Bad objective_val dimensions");
        io_bin(is, magicEof3);
        eof        = is.eof() || MAGIC_EQU(magicEof1,magicEof) != 0;
        if( ! eof ) throw runtime_error("MCsoln bad LONG data terminator");
    }catch(exception const& e){
        cout<<e.what()<<"\n\tRecovering: ignoring optional long-format data";
        // many long read errors are recoverable --- just resize as SHORT data.
        weights      .resize(0,0);
        lower_bounds .resize(0,0);
        upper_bounds .resize(0,0);
        objective_val.resize(0);
        //keep_going = 0;
    }
    return;
#undef IO_VEC
#undef IO_MAT
}

void testMCsolnWriteRead( MCsoln const& mcsoln, enum MCsoln::Fmt fmt, enum MCsoln::Len len)
{
    stringstream ss;
    ostream& os = ss;
    mcsoln.write( os, fmt, len );
    os.flush();
    MCsoln x;
    istream& is = ss;
    x.read( is );
    cout<<" testMCsolnWriteRead OK, XXX correctness tests TBD !!!"<<endl;
    return;
}
