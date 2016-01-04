
#include "mclua.hpp"

#include "base/app_state.hpp"
#include "script_lua/lua_interpreter.hpp"

using namespace std;
using namespace MILDE;

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
FUN(params)
;

// ------------------ scripting impl --------------------
namespace MILDE {

    int script_MCparm::f___gc() {
        scr_CNT;
        {   scr_USR( scr_MCparm, f, ERR_LBL );
            scr_STK( "<mcparm> : __gc()" );
            GAS.d_si->del_user( f );
            return 0;
        }
        scr_ERR( "<mcparm> : __gc()" );
    }

    int script_MCparm::f_type() ///< return string/name of the <mcparm>
    {
        scr_CNT;
        {   scr_USR( scr_MCparm, x, ERR_LBL );
            scr_STK( "<mcparm:type()-><str>" );
            GAS.d_si->put_ccstr( "mcparm" );
            return 1;
        }
        scr_ERR( "<mcparm:type()-><str>" );
    }

    int script_MCparm::f_get_no_projections()
    {
        scr_CNT;
        {   scr_USR( scr_MCparm, x, ERR_LBL );
            scr_STK( "<mcparm:type()-><int>" );
            GAS.d_si->put_int( x->d_params->no_projections );
            return 1;
        }
        scr_ERR( "<mcparm:type()-><int>" );
    }

    int script_MCparm::f_params()
    {
        scr_CNT;
        scr_MCparm(); 
        return 1;
    }


}//MILDE::


// ------------------- lua interface -------------------

static const struct luaL_Reg lua_mcparm_lib_m [] = {
    LUA_FUN(__gc),
    LUA_FUN(type),
    LUA_FUN(get_no_projections),
    {0,0}
};
static const struct luaL_Reg lua_mc_lib_f [] = {
    LUA_FUN(params)
};

extern "C" DLLEXP int luaopen_mcparm_base( lua_State *)
{
    MILDE_li()->register_class( OBJNAME(scr_MCparm), "mcparm", lua_mcparm_lib_m );
    // constructors
    MILDE_li()->register_namespace( NULL, "mc", lua_mc_lib_f );
    return 1;
}

