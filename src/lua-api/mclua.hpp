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

    /** lua interface functions for scr_MCparm object.
     *
     * Initialize lua namespace (script_MCparm)
     * ~~~{.lua}
     *   require('milde')
     *   mc = require('mcparm')
     * ~~~
     * Construct default parameters (+ print, convert to lua table)
     * ~~~{.lua}
     *   p = mc.new()       -- default parms
     *   print(p:str())     -- print: empty, because all values at default
     *   print(p:str(true)) -- print, verbose option
     *   t=p.get()          -- lua table (empty) of nondefault parms
     *   t=p.get(true)      -- lua table of all parms
     * ~~~
     * Modify parameters
     * ~~~{.lua}
     *   p = mc.new()
     *   t={}
     *   t.no_projections = 7.7777        -- int, ignores trailing .77777
     *   t.avg_epoch = 10                 -- size_t
     *   t.update_type = "MINIBATCH_SGD"  -- enum, using full value
     *   t.eta_type = "sqrt"              -- enum, using a lowercase substr
     *   t.C1 = 7.7777                    -- double, so .7777 not ignored
     *   t.resume = 1                     -- bool
     *
     *   p:set(t) -- <------- existing keys of p are overwritten by any settings in t
     *   print(p:str())    -- now have some nondefault parameters
     * ~~~
     *
     * Note: getargs and setargs use the normal milde version, base on \c Args
     *       but I prefer the get/set versions based on \c ArgMap.
     */
    struct script_MCparm {
        static int f___gc();
        static int f_type();
        // f_get may be a bit more convenient to use than getargs
        //    E.g., don't need tonumber(string) for assertions.
        static int f_get();     ///< string -> lua builtin type -- copy of MCfilter parms
        static int f_getargs(); ///< string -> string table -- copy of MCfilter parms
        static int f_set();     ///< set(<table>) overwrite parameter entries
        static int f_setargs(); ///< set(<table>) (string keys) overwrite parms
        static int f_str();     ///< str([verbose:bool=0]) nondefault (or all) parms
        // constructors
        static int f_new();  // return a default-constructed "param_struct"
    };

    /** Actually, should create a solver class that, besides \c solve_optimization,
     * provides write|read_binary|ascii of internal state.
     * - Internal state should be stored in short|full format
     *   - first store time t and time t state like eta, C1, C2, ...
     *   - then store
     *     - either short restart data from time-averaged {w,l,u,objective}
     *     - or long restart data that adds {w,l,u,obj} at time "t"
     */
    struct script_MCsolve {
        static int f___gc();
        /** Run an initialized solver, and then save restart data.
         * - Inputs describe where final data should be saved
         *  - [ rfile_avg               : restart filename for _avg data
         *  - [, rfile ]]               : restart filename for time t data
         */
        static int f_solve();
        /** construct a solver, possibly resuming an old calculation.
         * - Inputs:
         *      - repo, parms           : start from random conditions
         *      - [, rfile_avg          : and given restart = restart_avg
         *      - [, rfile ]]           : and, oh, use time t data instead
         *   - where rfile* are filenames for binary restart data
         * - restart data files contain:
         *   - DenseM weights
         *   - DenseM lower_bounds
         *   - DenseM upper_bounds
         *   - VectorXd objective_val
         * - and restart dimensions must agree with repo sizing.
         */     
        static int f_solver();
    };

}//MILDE::
#endif // MCLUA_H
