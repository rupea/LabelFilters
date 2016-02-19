
#include "mclua.hpp"
//#include "parameter-args.h"
//#include "mcsolveProg.hpp"

#include "base/app_state.hpp"
//#include "lua.h"
#include "lua.hpp"
#include "script_lua/lua_interpreter.hpp"
#include "base/argmap.hpp"
#include "repo/args.hpp"

#include <string.h>     // strtok
#include <assert.h>
#include <iostream>
#include <exception>

using namespace std;
using namespace MILDE;

#define tT template<typename T>

/** standard opening brace */
#define scr_TRY( ERrmsg ) char const* const errmsg = ERrmsg; scr_CNT; try

/** standard closing brace, with ERR_LBL: for goto's */
#define scr_CATCH catch(std::exception& e){ \
    cout<<" exception: "<<e.what(); \
    Derr_msg( false, true, e.what() ); \
    goto ERR_LBL; \
} \
scr_ERR( errmsg )

/** abbreviated lua stack check */
#define scr_CHK scr_STK(errmsg) 

// ------------------ helpers --------------------
/** convert std::string --> argc,argv at plain whitespace sequences */
struct ArgcArgv {
    ArgcArgv() : argc(0), argv(nullptr), buffer(nullptr) {}
    ~ArgcArgv(){
        delete[] buffer;
#ifndef NDEBUG
        for(int a=0; a<argc; ++a){ argv[a]=nullptr; }
        buffer=nullptr;
#endif
    }
    /** split s at whitepace sequences (plain strtok: no escape seqs, no quote handling).
     * esc seq could be handled within 's', perhaps.
     */
    ArgcArgv( std::string const s )
    //: argc(0U), argv(nullptr), nchar(s.size()), buffer( new char[nchar+1U] )
    : argc(0U), argv(nullptr), buffer(nullptr)
    {
#if 0 // OLD CODE : plain-old whitespace splitting via strtok
        // work on a copy of the input
        strncpy( buffer, s.data(), nchar );
        buffer[nchar] = '\0';
        // buffer --> tok[] ptrs
        char const delim[8] = " \t\n\r";        // did I miss some?
        char* bufptr = strtok(buffer,&delim[0]);
        while( bufptr ){
            tok.push_back(bufptr);
            bufptr = strtok(nullptr, delim);
        }
        // tok[] ptrs --> argc, argv
        argc = tok.size();
        tok.push_back(nullptr); // one extra nullptr, for strict argc/argv compat
        argv = &tok[0];
#endif
        std::vector<std::string> split = ::opt::cmdSplit(s,0); // 0 -> NO program-name, just args
        if( split.size() ){
            size_t nchar = 0U;
            for(auto const s: split) nchar += s.size() + 1U;
            buffer = new char[nchar];
            if( !buffer ) throw std::runtime_error("ArgcArgv out-of-mem");
            argv = new char*[split.size()+1U]; // one extra zero-length string
            if( !argv ) throw std::runtime_error("ArgcArgv out-of-mem");
            argc=0;
            size_t o=0U;
            for(size_t i=0U; i<split.size(); ++argc, ++i){
                argv[argc] = &buffer[o];
                memcpy( &buffer[o], split[i].data(), split[i].size() );
                o += split[i].size();
                buffer[o++] = '\0';
            }
            assert( argc == static_cast<int>(split.size()) );
            // and one extra nullptr
            argv[argc] = nullptr;
        }
    }
    int argc;
    char** argv;
private:
    //size_t const nchar;
    char* buffer;
    //std::vector<char*> tok;/
    //std::vector<std::string> split;
};

// ------------------ lua stack allocation --------------------
namespace MILDE {
    scr_MCparm::cnt_Params::cnt_Params()
        : param_struct( set_default_params() )
    {}

    // static relay functions to C++ constructor
    scr_MCparm* scr_MCparm::new_stack(){
        return new( GAS.d_si->new_user< scr_MCparm >( OBJNAME(scr_MCparm)))
            scr_MCparm;
    }

    scr_MCsolve* scr_MCsolve::new_stack( int argc, char**argv, param_struct const* const defparms/*=nullptr*/ ){
        return new( GAS.d_si->new_user< scr_MCsolve >( OBJNAME(scr_MCsolve)))
            scr_MCsolve( argc, argv, defparms );
    }

    scr_MCproj* scr_MCproj::new_stack( int argc, char**argv ){
        return new( GAS.d_si->new_user< scr_MCproj >( OBJNAME(scr_MCproj)))
            scr_MCproj( argc, argv );
    }

}//MILDE::

// ------------------- lua interface -------------------

#define FUN(name) WRAP_FUN(name,script_MCparm::f_)
FUN(__gc)
FUN(type)
FUN(pretty)
FUN(help)
FUN(get)        // return all parameters as string->luatype table
FUN(getargs)    // return all parameters as string->string table
FUN(set)        // overwrite keys in supplied string->luatype table
FUN(setargs)    // overwrite keys in supplied string->string table
FUN(str)
//FUN(load)       // read x,y training data
//FUN(xfile)      // set mcsolve/mcproj --xfile string
//FUN(yfile)      // set mcsolve/mcproj --yfile string
//FUN(xyfile)     // set --xfile and --yfile as fname.x and fname.y
//FUN(solnfile)   // set --solnfile string, and [def. "BS"] Binary|Text, Short|Long format

FUN(new)        // construct a default 'mcparm' parameter set
//FUN(toxy)    // toxy(repo,basename [,bool xnorm=false]) writes Eigen-friendly basename.{x|y} files usable for xfile and yfile commands
;
#undef FUN

/** to avoid name clash, modify WRAP_FUN naming convention */
#define WRAP_MCSOLVE(name,prefix) \
    static int luaB_solve_##name( lua_State* L ) \
{ \
    SWAP_LUA_STATE; \
    LUA_CHECK_FOR_DUMMY_ARG_HACK(name); \
    return prefix##name(); \
}
#define FUN(name) WRAP_MCSOLVE(name,script_MCsolve::s_)
FUN(__gc)
FUN(type)
FUN(help)
FUN(read)
FUN(solve)
FUN(save)
FUN(display)
//FUN(solve)   ... or read, solve, save, display
FUN(new)
;
#undef FUN

/** to avoid name clash, modify WRAP_FUN naming convention */
#define WRAP_MCPROJ(name,prefix) \
    static int luaB_project_##name( lua_State* L ) \
{ \
    SWAP_LUA_STATE; \
    LUA_CHECK_FOR_DUMMY_ARG_HACK(name); \
    return prefix##name(); \
}
#define FUN(name) WRAP_MCPROJ(name,script_MCproj::p_)
FUN(__gc)
FUN(type)
FUN(help)
FUN(read)
FUN(proj)
FUN(save)
FUN(validate)
//FUN(solve)   ... or read, solve, save, display
FUN(new)
;
#undef FUN

// ------------------ scripting impl --------------------
namespace MILDE {

    int script_MCparm::f___gc() {
        scr_TRY("<mcparm> : __gc()"){
            scr_USR( scr_MCparm, f, ERR_LBL );
            scr_STK( "<mcparm> : __gc()" );
            GAS.d_si->del_user( f );
            return 0;
        }scr_CATCH;
    }

    int script_MCparm::f_type() ///< return string/name of the <mcparm>
    {
        scr_TRY( "<mcparm>:type() -> <str>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            GAS.d_si->put_ccstr( "mcparm" );
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_pretty() ///< return string/name of the <mcparm>
    {
        scr_TRY( "<mcparm>:pretty() -> <str>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            string pretty;
            {
                ostringstream oss;
                oss << *x->d_params;
                pretty = oss.str();
            }
            GAS.d_si->put_ccstr( pretty );
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_help() ///< return string/name of the <mcparm>
    {
        scr_TRY( "<mcparm>.help() -> <str>" ){
            string help = ::opt::helpMcParms();
            GAS.d_si->put_ccstr( help );
            return 1;
        }scr_CATCH;
    }

#if 0
    // since I have no lua support for any 'raw' Eigen types, easier to supply filenames
    int script_MCparm::f_load() ///< loads an x,y dataset (repo)
    {
        scr_TRY( "<mc>.load(<fname:str>) -> sets x,y filenames to basename.{x|y}" ){
            scr_STR( fname, ERR_LBL );
            if(0){
                // read stuff HERE ...
                GAS.d_si->put_int( 1 );             // good read
                return 1;
            }else{
                std::ostringstream oss;
                oss<<"Error: <mc>:load( fname = "<<fname<<" ) failed\n";
                throw std::runtime_error(oss.str());
            }
        }scr_CATCH;
    }
#endif

    /** set \c param_struct::PARM to (MILDETYPE) args["PARM"] */
#define MCSET(MILDETYPE,PARM) do{ \
    if( args.map().find( #PARM ) != args.map().cend() ){ \
        MILDETYPE p; \
        args.get_##MILDETYPE( true/*abort*/, #PARM, p ); \
        x->d_params->PARM = p; \
    } \
}while(0)
    /** set, via CONV( (MILDETYPE)args["PARM"], \c param_struct::PARM ). */
#define MCSET_ENUM(CONV,PARM) do{ \
    cccstr* s = args.get( #PARM ); \
    if( s != nullptr ) { \
        string ss = *s; \
        CONV( ss, x->d_params->PARM ); \
    } \
}while(0)
    /** Any keys in ArgMap: string->int|dbl|bool|str replace any existing values */
    int script_MCparm::f_setargs()
    {
        scr_TRY( "<mcparm>:set({args}) -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            scr_ARGS( args, ERR_LBL );
            MCSET(int,no_projections);
            MCSET(real8,C1);
            MCSET(real8,C2);
            MCSET(uint8,max_iter);
            MCSET(uint8,batch_size);
            MCSET_ENUM(fromstring,update_type); // fromstring(args[key], enum&)
            MCSET(real8,eps);
            MCSET_ENUM(fromstring,eta_type); // string --> enum Eta_Type
            MCSET(real8,min_eta);
            MCSET(uint8,avg_epoch);
            MCSET(uint8,reorder_epoch);
            MCSET(uint8,report_epoch);
            MCSET(uint8,report_avg_epoch);
            MCSET(uint8,optimizeLU_epoch);
            MCSET(bool,remove_constraints);
            MCSET(bool,remove_class_constraints);
            MCSET_ENUM(fromstring,reweight_lambda);
            MCSET_ENUM(fromstring,reorder_type); // enum Reorder_Type
            MCSET(bool,ml_wt_by_nclasses);
            MCSET(bool,ml_wt_class_by_nclasses);
            MCSET(int,num_threads);
            MCSET(int,seed);
            MCSET(uint8,finite_diff_test_epoch);
            MCSET(uint8,no_finite_diff_tests);
            MCSET(real8,finite_diff_test_delta);
            MCSET(bool,resume);
            MCSET(bool,reoptimize_LU);
            MCSET(int,class_samples);
            return 0;
        }scr_CATCH;
    }
#undef MCSET_ENUM
#undef MCSET

#define MCSET(LUATYPE,CTYPE,CONV,PARM) do{ \
    if( /*bool*/argmap.map().find( #PARM ) ){ \
        CTYPE p; \
        argmap.get_##LUATYPE( string(#PARM), p, false/*err_if_missing*/ ); \
        x->d_params->PARM = CONV(p); \
    } \
}while(0)
/** This impl is compatibile with ArgMap. But has some deficiencies.
 * 
 * - mostly accepts only properly-typed lua objects
 *   - e.g. bool only accepts true/false lua value (0 or 1 won't work)
 * - enum as int value has been allowed.
 */
#define MCSET_enum(PARM) do{ \
    try{ \
        string s = argmap.map().get( #PARM ); \
        cout<<" argmap get("<<#PARM<<") --> enum string "<<s<<endl; \
        if( s.size() ) { \
            fromstring( s, x->d_params->PARM ); \
        } \
    }catch(...){ /* also allow an int value for the enum type */ \
        int i; \
        argmap.get_int( string(#PARM), i, false/*err_if_missing*/ ); \
        cout<<" argmap get_int("<<#PARM<<") --> enum int "<<i<<endl; \
        x->d_params->PARM = static_cast<decltype(x->d_params->PARM)>(i); \
    } \
}while(0)
#define MCSET_bool(PARM)   MCSET(bool,bool,          ,PARM)
#define MCSET_int(PARM)    MCSET(int, int,           ,PARM)
#define MCSET_size_t(PARM) MCSET(int, int,   (size_t),PARM)
#define MCSET_double(PARM) MCSET(dbl, double,        ,PARM)
    int script_MCparm::f_set()
    {
        scr_TRY( "<mcparm>:set({args}) -> <ArgMap_table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            {
                scr_ARGMAP( argmap, NOT_ARGS );
                MCSET_int(    no_projections );
                MCSET_double( C1 );
                MCSET_double( C2 );
                MCSET_size_t( max_iter );
                MCSET_size_t( batch_size );
                MCSET_enum(   update_type ); // tostring(enum Update_Type)
                MCSET_double( eps );
                MCSET_enum(   eta_type ); // enum Eta_Type
                MCSET_double( min_eta );
                MCSET_size_t( avg_epoch );
                MCSET_size_t( reorder_epoch );
                MCSET_size_t( report_epoch );
                MCSET_size_t( report_avg_epoch );
                MCSET_size_t( optimizeLU_epoch );
                MCSET_bool(   remove_constraints );
                MCSET_bool(   remove_class_constraints );
                MCSET_enum(   reweight_lambda );
                MCSET_enum(   reorder_type ); // enum Reorder_Type
                MCSET_bool(   ml_wt_by_nclasses );
                MCSET_bool(   ml_wt_class_by_nclasses );
                MCSET_int(    num_threads );
                MCSET_int(    seed );
                MCSET_size_t( finite_diff_test_epoch );
                MCSET_size_t( no_finite_diff_tests );
                MCSET_double( finite_diff_test_delta );
                MCSET_bool(   resume );
                MCSET_bool(   reoptimize_LU );
                MCSET_int(    class_samples );
                return 1;       // returns some table
            }
NOT_ARGS:
            {
                scr_STR( cmdline, ERR_LBL );
                scr_STK("<mcparm>:set(<cmd_line_args: str>) -> <this:mcparm> -- use cmdline to modify existing MCsolver parameters, see also mcparm.help()");
                ArgcArgv aa(cmdline);                   // tokens separated by [[:white:]]*
                // GOOD: x fully initialized before this call (by scr_MCparm constructor)
                auto unparsed = ::opt::mcArgs( aa.argc, aa.argv, *(x->d_params) );
                if( unparsed.size() ) cerr<<" (ignoring "<<unparsed.size()<<" unparsed args for now)"<<endl;
                // TODO also return 'unparsed' as a string table [0,1,2,...] --> lua string
                return 1;
            }
        }scr_CATCH;
    }
#undef MCSET_double
#undef MCSET_size_t
#undef MCSET_int
#undef MCSET_bool
#undef MCSET_enum
#undef MCSET


/** TYPE=bool|int|real4|<any milde type>, PARM=item in \c param_struct.
 * Note that MILDE::Args stores all keys as <string>. */
#define MCARGS(TYPE,PARM) do { \
    bool const diff = x->d_params->PARM != def.PARM; \
    if( all || diff ) \
        p.set( #PARM, (TYPE)(x->d_params->PARM) ); \
}while(0)

    int script_MCparm::f_getargs()
    {
        scr_TRY( "<mcparm>:get([all:bool=false]) -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            bool all = false;
            {   scr_BOOL( a, GOT_ALL );
                scr_STK("<mcparm>:str( <all:bool> ) -> <cccstr>");
                all = a;
            }
GOT_ALL:
            param_struct def = set_default_params();
            Args p;      // --> all keys will actually be strings, MILDE types
            // ArgMap p; // --> all keys will be closes lua base type, LUA types
            MCARGS(uint4,no_projections);
            MCARGS(real8,C1);
            MCARGS(real8,C2);
            MCARGS(uint8,max_iter);
            MCARGS(uint8,batch_size);
            MCARGS(int,update_type); // enum Update_Type
            MCARGS(real8,eps);
            MCARGS(int,eta_type); // enum Eta_Type
            MCARGS(real8,min_eta);
            MCARGS(uint4,avg_epoch);
            MCARGS(uint4,reorder_epoch);
            MCARGS(uint4,report_epoch);
            MCARGS(uint4,report_avg_epoch);
            MCARGS(uint4,optimizeLU_epoch);
            MCARGS(bool,remove_constraints);
            MCARGS(bool,remove_class_constraints);
            MCARGS(int,reweight_lambda);
            MCARGS(int,reorder_type); // enum Reorder_Type
            MCARGS(bool,ml_wt_by_nclasses);
            MCARGS(bool,ml_wt_class_by_nclasses);
            MCARGS(int,num_threads);
            MCARGS(int,seed);
            MCARGS(uint4,finite_diff_test_epoch);
            MCARGS(uint4,no_finite_diff_tests);
            MCARGS(real8,finite_diff_test_delta);
            MCARGS(bool,resume);
            MCARGS(bool,reoptimize_LU);
            MCARGS(int,class_samples);
            GAS.d_si->put_stack( p );
            return 1;
        }scr_CATCH;
    }
#undef MCARGS

/** TYPE=bool|dbl|int|str, XFORM function applied to retrieved PARM in \c param_struct.
 * Note that MILDE::ArgMap stores all keys as some base <lua type>. */
#define MCARGS(TYPE,XFORM,PARM) do { \
    bool const diff = x->d_params->PARM != def.PARM; \
    if( all || diff ) \
        p.set_ ## TYPE ( #PARM, XFORM(x->d_params->PARM)); \
}while(0)

    int script_MCparm::f_get()
    {
        scr_TRY( "<mcparm>:get([all:bool=false]) -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            bool all = false;
            {   scr_BOOL( a, GOT_ALL );
                scr_STK("<mcparm>:get( <all:bool> ) -> <table:string->various>");
                all = a;
            }
GOT_ALL:
            param_struct def = set_default_params();
            // Args p; // --> all keys will actually be strings, MILDE types
            ArgMap p;  // --> all keys will be closes lua base type, LUA types
            MCARGS(int,,no_projections);
            MCARGS(dbl,,C1);
            MCARGS(dbl,,C2);
            MCARGS(int,,max_iter);
            MCARGS(int,,batch_size);
            MCARGS(str,tostring,update_type); // tostring(enum Update_Type)
            MCARGS(dbl,,eps);
            MCARGS(str,tostring,eta_type); // enum Eta_Type
            MCARGS(dbl,,min_eta);
            MCARGS(int,,avg_epoch);
            MCARGS(int,,reorder_epoch);
            MCARGS(int,,report_epoch);
            MCARGS(int,,report_avg_epoch);
            MCARGS(int,,optimizeLU_epoch);
            MCARGS(bool,,remove_constraints);
            MCARGS(bool,,remove_class_constraints);
            MCARGS(int,,reweight_lambda);
            MCARGS(str,tostring,reorder_type); // enum Reorder_Type
            MCARGS(bool,,ml_wt_by_nclasses);
            MCARGS(bool,,ml_wt_class_by_nclasses);
            MCARGS(int,,num_threads);
            MCARGS(int,,seed);
            MCARGS(int,,finite_diff_test_epoch);
            MCARGS(int,,no_finite_diff_tests);
            MCARGS(dbl,,finite_diff_test_delta);
            MCARGS(bool,,resume);
            MCARGS(bool,,reoptimize_LU);
            MCARGS(int,,class_samples);
            GAS.d_si->put_stack( p );
            return 1;
        }scr_CATCH;
    }
#undef MCARGS

    int script_MCparm::f_str()
    {
        scr_TRY( "<mcparm>:str([verbose:bool=false]) -> <string> -- return pretty table of values" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            bool verbose = false;
            {   scr_BOOL( v, GOT_VERBOSE );
                scr_STK("<mcparm>:str( <verbose:bool> ) -> <cccstr>");
                verbose = v;
            }
GOT_VERBOSE:
            assert( x->d_params.valid() );
            string s("BAD scr_MCparm");
            if( x->d_params.valid() ){
                param_struct def = set_default_params();
                uint32_t nNonDefault = 0U;
                ostringstream oss;
                oss<<" MCFilter param_struct: ";
                if( verbose ) oss<<"(* = non-default)";
                oss<<"\n";
#define MCPARM(W,PARM,MSG) do { \
    bool const diff = x->d_params->PARM != def.PARM; \
    char const* defmark = (verbose? (diff? "* ":""): "* "); \
    if(diff) ++nNonDefault; \
    if(verbose || diff) { \
        oss <<setw(W)<< #PARM <<setw(30-W)<<defmark \
        <<    setw(15)<<left \
        <<(x->d_params->PARM) \
        <<right; \
        if(verbose) oss<<" "<<MSG; \
        oss<<"\n"; \
    }}while(0)
                MCPARM(20,no_projections,"number of projections to be made");
                MCPARM(20,C1,"penalty for example outside its class boundary");
                MCPARM(20,C2,"penalty for example inside other class's boundary");
                MCPARM(20,max_iter,"max # iterations");
                MCPARM(20,batch_size,"minibatch size");
                MCPARM(20,update_type,"how to update w, L and U");
                MCPARM(20,eps,"not used");
                MCPARM(20,eta_type,"learning rate decay schedule");
                MCPARM(25,eta,"initial learning rate");
                MCPARM(25,min_eta,"min value of learning rate");
                MCPARM(20,avg_epoch,"iteration at which avg'ing starts");
                MCPARM(20,reorder_epoch,"iterations between class reorderings");
                MCPARM(20,report_epoch,"iterations between report of objective value (over ENTIRE training set)");
                MCPARM(20,report_avg_epoch,"iterations between report of obj val for avg_w (over ENTIRE training set)");
                MCPARM(20,optimizeLU_epoch,"iterations between full optimization of upper and lower bounds");
                MCPARM(27,remove_constraints,"bool: remove for instances outside bounds of previous projections");
                MCPARM(27,remove_class_constraints,"bool: remove for examples outside own class bounds in prev projections");
                MCPARM(20,reweight_lambda,"lower lambda (increase C1 & C2) as constraints are eliminated");
                MCPARM(20,reorder_type,"rank by projected means of boundaries of prev projections");
                MCPARM(27,ml_wt_by_nclasses,"bool: other constraints weight examples by # classes example belongs to");
                MCPARM(27,ml_wt_class_by_nclasses,"bool: self constraints weight examples by # classes example belongs to");
                MCPARM(20,num_threads,"perhaps use OMP_NUM_THREADS instead?");
                MCPARM(20,seed,"for srand");
                MCPARM(20,finite_diff_test_epoch,"gradient correctness test interval");
                MCPARM(20,no_finite_diff_tests,"# rand examples tested for gradient correctness during each test");
                MCPARM(20,finite_diff_test_delta,"threshold for gradient correctness test");
                MCPARM(20,resume,"bool: train more projections?");
                MCPARM(20,reoptimize_LU,"bool: reoptimize class bounds");
                MCPARM(20,class_samples,"number of negative classes to use at each gradient iteration [0=all]");

#undef MCPARM
                if( nNonDefault == 0U )
                    oss<<"          All values at default settings";
                s = oss.str();
            }
            GAS.d_si->put_ccstr(s);
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_new()
    {
        scr_TRY("<mcparm>.new() -> <mcparm>"){
            // OPT: f_get of default settings returns empty table
            //return script_MCparm::f_get();
            //ScrAPI_create_object( script_mcparm, script_MCparm, "params" );
            //scr_MCparm *x =
            scr_MCparm::new_stack(); 
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_toxy()
    {
        scr_TRY("<mcparm>.toxy( <repo>, <basename:str> ) -- write basename.x basename.y data files"){
            cerr<<"XXX TBD f_toxy"<<endl;
            throw std::runtime_error(" script_MCparm::f_toxy NOT IMPLEMENTED");
            //return 1;
        }scr_CATCH;
    }

    int script_MCsolve::s___gc() {
        scr_TRY("<mcsolve> : __gc()"){
            scr_USR( scr_MCsolve, f, ERR_LBL );
            scr_STK( "<mcsolve> : __gc()" );
            GAS.d_si->del_user( f );
            return 0;
        }scr_CATCH;
    }

    int script_MCsolve::s_type() ///< return string/name of the <mcsolve>
    {
        scr_TRY( "<mcsolve>:type() -> <str>" ){
            scr_USR( scr_MCsolve, x, ERR_LBL );
            GAS.d_si->put_ccstr( "mcsolve" );
            return 1;
        }scr_CATCH;
    }

    int script_MCsolve::s_help() ///< return string/name of the <mcparm>
    {
        scr_TRY( "<mcsolve>.help() -> <str>" ){
            string help = ::opt::MCsolveArgs::defaultHelp();
            help.append( "lua constructors:\n"
                         "    - <mcsolve>.new(\"--xfile=... --yfile=... --solnfile=... etc\")\n"
                         "    - <mcsolve>.new(<explicit_defaults:xparm>, \"--xfile=... etc\")\n"
                         " Program default settings can come from:\n"
                         "   1. lua <explicit_defaults:xparm>\n"
                         "   2. else parameters from previous --solnfile\n"
                         "   3. else library default settings\n"
                       );
            GAS.d_si->put_ccstr( help );
            return 1;
        }scr_CATCH;
    }

    int script_MCsolve::s_read()
    {
        scr_TRY( "<mcsolve>:read() -> <err:bool>" ){
            scr_USR( scr_MCsolve, x, ERR_LBL );
            x->d_solve->tryRead(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCsolve::s_solve()
    {
        scr_TRY( "<mcsolve>:solve() -> <err:bool>" ){
            scr_USR( scr_MCsolve, x, ERR_LBL );
            x->d_solve->trySolve(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCsolve::s_save()
    {
        scr_TRY( "<mcsolve>:save() -> <err:bool>" ){
            scr_USR( scr_MCsolve, x, ERR_LBL );
            x->d_solve->trySave(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCsolve::s_display()
    {
        scr_TRY( "<mcsolve>:display() -> <err:bool>" ){
            scr_USR( scr_MCsolve, x, ERR_LBL );
            x->d_solve->tryDisplay(/*verbose*/);
            return 1;
        }scr_CATCH;
    }

    int script_MCsolve::s_new()
    {
        scr_TRY("   <mcsolve>.new(<cmdLineArgs:str>) -> <mcsolve>"
                "\nOR <mcsolve>.new(<mcparm>, <cmdLineArgs:str>) -> <mcsolve>"
               ){
            param_struct *explicit_defaults = nullptr;
            ::opt::MCsolveArgs mcsa;   // temp copy, real one inside MCsolveProgram
            {
                scr_USR( scr_MCparm, defaultParms, NO_PARMS );
                explicit_defaults = defaultParms->d_params;
            }
NO_PARMS:
            scr_STR( cmdLineArgs, ERR_LBL );
            ArgcArgv aa(cmdLineArgs);
            scr_MCsolve::new_stack( aa.argc, aa.argv, explicit_defaults );
            return 1;
        }scr_CATCH;
    }

    // ------------------ projection code -----------------------
    int script_MCproj::p___gc() {
        scr_TRY("<mcproj> : __gc()"){
            scr_USR( scr_MCproj, f, ERR_LBL );
            scr_STK( "<mcproj> : __gc()" );
            GAS.d_si->del_user( f );
            return 0;
        }scr_CATCH;
    }

    int script_MCproj::p_type() ///< return string/name of the <mcproj>
    {
        scr_TRY( "<mcproj>:type() -> <str>" ){
            scr_USR( scr_MCproj, x, ERR_LBL );
            GAS.d_si->put_ccstr( "mcproj" );
            return 1;
        }scr_CATCH;
    }

    int script_MCproj::p_help() ///< return string/name of the <mcparm>
    {
        scr_TRY( "<mcproj>.help() -> <str>" ){
            string help = ::opt::MCprojArgs::defaultHelp();
            help.append( "lua constructors:\n"
                         "    - <mcproj>.new(\"--solnfile=... --xfile=... etc\")\n"
                       );
            GAS.d_si->put_ccstr( help );
            return 1;
        }scr_CATCH;
    }

    int script_MCproj::p_read()
    {
        scr_TRY( "<mcproj>:read() -> <err:bool>" ){
            scr_USR( scr_MCproj, x, ERR_LBL );
            x->d_proj->tryRead(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCproj::p_proj()
    {
        scr_TRY( "<mcproj>:proj() -> <err:bool>" ){
            scr_USR( scr_MCproj, x, ERR_LBL );
            x->d_proj->tryProj(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCproj::p_save()
    {
        scr_TRY( "<mcproj>:save() -> <err:bool>" ){
            scr_USR( scr_MCproj, x, ERR_LBL );
            x->d_proj->trySave(/*verbose*/);
            return 1;
        }scr_CATCH;
    }
    int script_MCproj::p_validate()
    {
        scr_TRY( "<mcproj>:display() -> <err:bool>" ){
            scr_USR( scr_MCproj, x, ERR_LBL );
            x->d_proj->tryValidate(/*verbose*/);
            return 1;
        }scr_CATCH;
    }


    int script_MCproj::p_new()
    {
        scr_TRY("   <mcproj>.new(<cmdLineArgs:str>) -> <mcproj>"
               ){
            scr_STR( cmdLineArgs, ERR_LBL );
            ArgcArgv aa(cmdLineArgs);
            scr_MCproj::new_stack( aa.argc, aa.argv );
            return 1;
        }scr_CATCH;
    }
}//MILDE::


// ------------------- lua interface -------------------

static const struct luaL_Reg lua_mcparm_lib_m [] = {
    LUA_FUN(__gc),
    LUA_FUN(type),
    LUA_FUN(pretty),
    LUA_FUN(help),
    LUA_FUN(get),
    LUA_FUN(getargs),
    LUA_FUN(set),
    LUA_FUN(setargs),
    LUA_FUN(str),
    //LUA_FUN(load),
    {0,0}
};
static const struct luaL_Reg lua_mc_lib_f [] = {
    LUA_FUN(new),
    {0,0}
};
#undef LUA_FUN
#define LUA_FUN(name) { #name , luaB_solve_##name }
static const struct luaL_Reg lua_mcsolve_lib_m [] = {
    LUA_FUN(__gc),
    LUA_FUN(type),
    LUA_FUN(help),
    LUA_FUN(read),
    LUA_FUN(solve),
    LUA_FUN(save),
    LUA_FUN(display),
    {0,0}
};
static const struct luaL_Reg lua_mcsolve_lib_f [] = {
    LUA_FUN(new),
    LUA_FUN(help),
    {0,0}
};
#undef LUA_FUN
#define LUA_FUN(name) { #name , luaB_project_##name }
static const struct luaL_Reg lua_mcproj_lib_m [] = {
    LUA_FUN(__gc),
    LUA_FUN(type),
    LUA_FUN(help),
    LUA_FUN(read),
    LUA_FUN(proj),
    LUA_FUN(save),
    LUA_FUN(validate),
    {0,0}
};
static const struct luaL_Reg lua_mcproj_lib_f [] = {
    LUA_FUN(new),
    LUA_FUN(help),
    {0,0}
};
#undef LUA_FUN

/** needs libraries milde_core and mcfilter.
 * Usage from within milde \em lua_cpp :
~~~{.lua}
  require("milde")
  -- create a lua mcparm object
  -- auto-invokes libmclua.mcparm during luaopen_libmclua, so remember return value
  mc=require("libmclua")
  --mc = libmclua.mc.new()     -- the only function in namespace libmclua.mcparm, now auto-invoked

  -- enum consts can be used, though, as:
  print(libmclua.ETA_LIN)                       -- enum value '2', see parameter.h
  for k,v in pairs(libmclua) do print(k,v) end  -- print out enums (and 'mc' subtable with 'mc.new()')

  -- use the lua mcparm object
  print(mc:type())
  print(mc:str())
  print(mc:str(true))                           -- verbose listing (a handy reference)
  tabArgs=mc:getargs()                          -- get all NON-DEFAULT args as string table (milde \b Args)
  for k,v in pairs(tabArgs) do print(k,v) end   -- this will be empty
  tabArgs=mc:getargs(true)                      -- get ALL args as table:str->str (milde \b Args)
  for k,v in pairs(tabArgs) do print(k,v) end   -- now non-empty

  -- I like this one a little better (uses milde \b ArgMap interface)
  tab=mc:get(true)                              -- get ALL args as table:str->lua_type (milde \b ArgMap)
  for k,v in pairs(tab) do print(k,v) end       -- named enums, and nicer printout for native lua values

  mc:set({no_projections=4,update_type="SAFE_SGD"})     -- set new values for 2 fields
  print(mc:str())                                       -- print out shows the 2 non-default values
  -- TODO accept either "SAFE_SGD" or numerical libmclua.SAFE_SGD for enum types ?
~~~
\sa script_MCparm
 */
extern "C" DLLEXP int luaopen_libmclua( lua_State * L )
{
    MILDE_li()->register_class( OBJNAME(scr_MCparm), "mcparm", lua_mcparm_lib_m );
    // constructors
    //   put table 'mc' into table 'libmclua'
    MILDE_li()->register_namespace( "libmclua", "mc", lua_mc_lib_f );

    MILDE_li()->register_class( OBJNAME(scr_MCsolve), "mcsolve", lua_mcsolve_lib_m );
    MILDE_li()->register_namespace( "libmclua", "mcsolve", lua_mcsolve_lib_f );

    MILDE_li()->register_class( OBJNAME(scr_MCproj), "mcproj", lua_mcproj_lib_m );
    MILDE_li()->register_namespace( "libmclua", "mcproj", lua_mcproj_lib_f );

    script_MCparm::f_new();             // automagically invoke libmclua.mc.new()

    // now you can:
    //    require("libmclua");
    //    mcsolv = libmclua.mcsolve.new("--maxiter=5000, --xfile=foo.x --yfile=foo.y --solnfile=foo.soln")
    //    ...

    lua_getglobal(L,"libmclua"); // get the table
    // ... or even lua_getlobal(MILDE_li()->d_lua,"libmclua");
    // push some handy constants...
#define ADD_ENUM( ENUM ) do {\
    lua_pushnumber(L, ENUM); \
    lua_setfield(L,-2,#ENUM); \
}while(0)
    ADD_ENUM(ETA_CONST);        // modify the table ... lua now has "libmclua.ETA_CONST"
    ADD_ENUM(ETA_SQRT);
    ADD_ENUM(ETA_LIN);
    ADD_ENUM(ETA_3_4);
    ADD_ENUM(MINIBATCH_SGD);
    ADD_ENUM(SAFE_SGD);
    ADD_ENUM(REORDER_AVG_PROJ_MEANS);
    ADD_ENUM(REORDER_PROJ_MEANS);
    ADD_ENUM(REORDER_RANGE_MIDPOINTS);
    ADD_ENUM(REWEIGHT_NONE);
    ADD_ENUM(REWEIGHT_LAMBDA);
    ADD_ENUM(REWEIGHT_ALL);
    lua_pop(L,1);               // pop the table
    return 1;
}

// alternate names for luaopen:
/** you can provide a link mcparm.so --> one of libmclua.so or libmclua-dbg.so */
extern "C" DLLEXP int luaopen_mcparm( lua_State *L )
{
    return luaopen_libmclua(L);
}

