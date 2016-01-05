
#include "mclua.hpp"

#include "base/app_state.hpp"
#include "script_lua/lua_interpreter.hpp"
#include "base/argmap.hpp"

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
FUN(get_no_projections)
FUN(get)        // return all parameters as string->double table
FUN(set)        // overwrite keys in supplied string->double table
FUN(str)

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

    int script_MCparm::f_get_no_projections()
    {
        scr_TRY( "<mcparm>:get_no_projections()-><int>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            GAS.d_si->put_int( x->d_params->no_projections );
            return 1;
        }scr_CATCH;
    }

    int script_MCparm::f_set()
    {
        scr_TRY( "<mcparm>:set({args}) -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            return 1;
        }scr_CATCH;
    }

    /** TYPE=bool|dbl|int|str, KEY=item in \c param_struct, DEF=default param_struct. */
#define ARGSET(MC,ARGMAP,TYPE,KEY,DEF) do{ \
    if( MC->d_params->KEY != DEF.KEY ){ \
        ARGMAP.set_##TYPE( #KEY , d_params->KEY ); \
    }}
    int script_MCparm::f_get()
    {
        scr_TRY( "<mcparm>:get() -> <table:string->various>" ){
            scr_USR( scr_MCparm, x, ERR_LBL );
            param_struct def = set_default_params();
            ArgMap p;
            if( x->d_params->no_projections != def.no_projections ){
                p.set_int("no_projections", x->d_params->no_projections);
            }
            GAS.d_si->put_stack( p );
            return 1;
        }scr_CATCH;
    }
#undef ARGSET

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
    bool diff = x->d_params->PARM != def.PARM; \
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
    LUA_FUN(get_no_projections),
    LUA_FUN(get),
    LUA_FUN(set),
    LUA_FUN(str),
    {0,0}
};
static const struct luaL_Reg lua_mc_lib_f [] = {
    LUA_FUN(new)
};

extern "C" DLLEXP int luaopen_mcparm( lua_State *)
{
    MILDE_li()->register_class( OBJNAME(scr_MCparm), "mcparm", lua_mcparm_lib_m );
    // constructors
    MILDE_li()->register_namespace( "script_MCparm", "mc", lua_mc_lib_f );
    return 1;
}

