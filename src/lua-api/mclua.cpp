
#include "mclua.hpp"
#include "lua.hpp"

#include "base/app_state.hpp"
#include "script_lua/lua_interpreter.hpp"
#include "base/argmap.hpp"
#include "repo/args.hpp"

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

// ------------------ lua stack allocation --------------------
namespace MILDE {
    scr_MCparm::cnt_Params::cnt_Params()
        : param_struct( set_default_params() )
    {}

    scr_MCparm* scr_MCparm::new_stack(){
        return new( GAS.d_si->new_user< scr_MCparm >( OBJNAME(scr_MCparm))) scr_MCparm;
    }
    //scr_MCparm foo(nullptr);
}//MILDE::

// ------------------- lua interface -------------------

#define FUN(name) WRAP_FUN(name,script_MCparm::f_)
FUN(__gc)
FUN(type)
FUN(get)        // return all parameters as string->luatype table
FUN(getargs)    // return all parameters as string->string table
FUN(set)        // overwrite keys in supplied string->luatype table
FUN(setargs)    // overwrite keys in supplied string->string table
FUN(str)
FUN(load)       // read x,y training data

FUN(new)     // construct a default 'mcparm' parameter set
;

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
        scr_TRY( "<mcparm>:type()-><str>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            GAS.d_si->put_ccstr( "mcparm" );
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_load() ///< loads an x,y dataset (repo)
    {
        scr_TRY( "<mc>.load(<fname:str>)-><str>" ){
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
#define MCSET_enum(PARM) do{ \
    string s = argmap.map().get( #PARM ); \
    cout<<" argmap get("<<#PARM<<") --> string "<<s<<endl; \
    if( s.size() ) { \
        fromstring( s, x->d_params->PARM ); \
    } \
}while(0)
#define MCSET_bool(PARM)   MCSET(bool,bool,          ,PARM)
#define MCSET_int(PARM)    MCSET(int, int,           ,PARM)
#define MCSET_size_t(PARM) MCSET(int, int,   (size_t),PARM)
#define MCSET_double(PARM) MCSET(dbl, double,        ,PARM)
    int script_MCparm::f_set()
    {
        scr_TRY( "<mcparm>:set({args}) -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            scr_ARGMAP( argmap, ERR_LBL );
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
            return 1;
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

}//MILDE::


// ------------------- lua interface -------------------

static const struct luaL_Reg lua_mcparm_lib_m [] = {
    LUA_FUN(__gc),
    LUA_FUN(type),
    LUA_FUN(get),
    LUA_FUN(getargs),
    LUA_FUN(set),
    LUA_FUN(setargs),
    LUA_FUN(str),
    LUA_FUN(load),
    {0,0}
};
static const struct luaL_Reg lua_mc_lib_f [] = {
    LUA_FUN(new)
};

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
    // lua stack: table named 'libmclua', with one entry, 'libmclua.mc'
    script_MCparm::f_new();             // automagically invoke libmclua.mc.new()

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

