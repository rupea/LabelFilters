#ifndef MCLUA_H
#define MCLUA_H

#include "base/rc_base.hpp"
#include "base/app_state.hpp"
//#include "script_lua/lua_binding_macros.hpp"
//#include "script_lua/lua_interpreter.hpp"
//#include "script_shell/script_stream.hpp"
#include "parameter.h"

//struct param_struct;

namespace MILDE {

    struct scr_MCparm{
        /** a ref-counted wrapper for a libmcfilter param_struct */
        struct cnt_Params :
            //public Base_tag,   // defined in milde lib/os/util_io.hpp
            public rc_Cnt,
            public param_struct
        {
            typedef param_struct base;
            cnt_Params();
            cnt_Params( base const& x ) : param_struct(x) {}
        };//class cnt_Params
        typedef rc_Ptr<cnt_Params> ptr_Params;

        explicit scr_MCparm() : d_params() {}
        virtual ~scr_MCparm() {}

        static scr_MCparm* new_stack(); // default-construct a param_struct

        ptr_Params d_params;
    };

    struct script_MCparm {
        static int f___gc();
        static int f_type();
        static int f_get_no_projections();
        // constructors
        static int f_params();  // return a default-constructed "param_struct"
    };

}//MILDE::
#endif // MCLUA_H
