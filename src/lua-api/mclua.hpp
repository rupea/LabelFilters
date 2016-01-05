#ifndef MCLUA_H
#define MCLUA_H

#include "base/rc_base.hpp"
#include "base/app_state.hpp"
#include "lua.h"
//#include "script_lua/lua_binding_macros.hpp"
//#include "script_lua/lua_interpreter.hpp"
//#include "script_shell/script_stream.hpp"
#include "parameter.h"

//struct param_struct;
extern "C" int luaopen_mcparm( lua_State *);

namespace MILDE {

    struct scr_MCparm{
        /** a ref-counted wrapper for a libmcfilter param_struct */
        struct cnt_Params :
            //public Base_tag,   // defined in milde lib/os/util_io.hpp
            public rc_Cnt,
            public param_struct
        {
            typedef param_struct base;
            cnt_Params();               ///< init via \c set_default_params().
            cnt_Params( base const& x ) : param_struct(x) {}
        };//class cnt_Params

        explicit scr_MCparm() : d_params(new cnt_Params()) {}
        virtual ~scr_MCparm() {}

        static scr_MCparm* new_stack(); ///< construct as lua userdata object

        rc_Ptr<cnt_Params> d_params;
    };

    struct script_MCparm {
        static int f___gc();
        static int f_type();
        static int f_get_no_projections();
        static int f_get();
        static int f_set();
        static int f_str();
        // constructors
        static int f_new();  // return a default-constructed "param_struct"
    };

}//MILDE::
#endif // MCLUA_H
