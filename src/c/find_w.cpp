
#include "find_w.hh"            // get template definition of the solve_optimization problem
#include "mcsolver.hh"          // template impl of MCsolver version
#include <stdexcept>
#include <fstream>
#include <sstream>

using namespace std;

// Explicitly instantiate templates into the library

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds,
                        VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg,
                        VectorXd& objective_val_avg,
                        const DenseM& x,                // <-------- EigenType
                        const SparseMb& y,
                        const param_struct& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds,
                        VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg,
                        VectorXd& objective_val_avg,
                        const SparseM& x,                // <-------- EigenType
                        const SparseMb& y,
                        const param_struct& params);

// Complete some of the class declarations before instantiating MCsolver
MCsolver::MCsolver(char const* const solnfile /*= nullptr*/)
: MCsoln()
    // private "solve" variables here TODO
{
    if( solnfile ){
        ifstream ifs(solnfile);
        if( ifs.good() ) try{
            this->read( ifs );
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
#define MAGIC_U32( ARRAY4C ) (*reinterpret_cast<uint_least32_t const*>( ARRAY4C.cbegin() ))
#define MAGIC_EQU( A, B ) (MAGIC_U32(A) == MAGIC_U32(B))

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
        IO_MAT(os, weights_avg , d     , nProj, "Bad weights_avg dimensions");
        IO_MAT(os, lower_bounds_avg, nClass, nProj, "Bad weights_avg dimensions");
        IO_MAT(os, upper_bounds_avg, nClass, nProj, "Bad weights_avg dimensions");
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
        IO_MAT(os, weights      , d     , nProj, "Bad weights dimensions");
        IO_MAT(os, lower_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds , nClass, nProj, "Bad lower_bounds dimensions");
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
        io_txt(is,magicData);
        if( ! MAGIC_EQU(magicData,magicCnt) )
            throw runtime_error("MCsoln header length has changed");
    }catch(exception const& e){
        cout<<e.what();
        throw std::runtime_error("MCsoln header data read error");
    }
    int keep_going, eof;
#define IO_MAT( IS, MAT, R, C, ERRMSG ) do{eigen_io_txt(IS,MAT); CHK_MAT_DIM( MAT,R,C,ERRMSG );}while(0)
#define IO_VEC( IS, VEC, SZ, ERRMSG ) do {eigen_io_txt(IS,VEC); CHK_VEC_DIM( VEC,SZ,ERRMSG );}while(0)
    try{
        do{
            eigen_io_txt(is,weights_avg);
            CHK_MAT_DIM( weights_avg,0,0,"bogus" );
        }while(0);
        IO_MAT(is, weights_avg , d     , nProj, "Bad weights_avg dimensions");
        IO_MAT(is, lower_bounds, nClass, nProj, "Bad weights_avg dimensions");
        IO_MAT(is, upper_bounds, nClass, nProj, "Bad weights_avg dimensions");
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
        IO_MAT(is, weights      , d     , nProj, "Bad weights dimensions");
        IO_MAT(is, lower_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_MAT(is, upper_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_VEC(is, objective_val, nClass, "Bad objective_val dimensions");
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
        IO_MAT(os, weights_avg , d     , nProj, "Bad weights_avg dimensions");
        IO_MAT(os, lower_bounds, nClass, nProj, "Bad weights_avg dimensions");
        IO_MAT(os, upper_bounds, nClass, nProj, "Bad weights_avg dimensions");
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
        IO_MAT(os, weights      , d     , nProj, "Bad weights dimensions");
        IO_MAT(os, lower_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_MAT(os, upper_bounds , nClass, nProj, "Bad lower_bounds dimensions");
        IO_VEC(os, objective_val, nClass, "Bad objective_val dimensions");
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
        ::read_ascii(is,parms);
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
        do{
            eigen_io_bin(is,weights_avg);
            CHK_MAT_DIM( weights_avg,0,0,"bogus" );
        }while(0);
        IO_MAT(is, weights_avg , d     , nProj, "Bad weights_avg dimensions");
        IO_MAT(is, lower_bounds, nClass, nProj, "Bad weights_avg dimensions");
        IO_MAT(is, upper_bounds, nClass, nProj, "Bad weights_avg dimensions");
        io_bin(is,magicEof1);
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
        IO_MAT(is, weights      , d     , nProj, "Bad weights dimensions");
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
