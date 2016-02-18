#ifndef MCLUA_H
#define MCLUA_H

#include "parameter.h"
#include "mcsolveProg.hpp"         // MCsolveProgram
#include <omp.h>                // MUST be included before milde XXX evil!

#include "base/rc_base.hpp"
//#include "base/app_state.hpp"
#include "lua.hpp"                // lua_State, from $(MILDE_DIR)/include/milde_lua
//#include "script_lua/lua_binding_macros.hpp"
//#include "script_lua/lua_interpreter.hpp"
//#include "script_shell/script_stream.hpp"
//#include <omp.h>                // clash for omp_get_num_threads, omp_test_lock, mutexlock stuff, etc.

//struct param_struct;
extern "C" int luaopen_mcparm( lua_State *);

namespace MILDE {

    /** internal C++ impl for param_struct MCfilter functionality */
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

        static scr_MCparm* new_stack(); ///< construct 'parms' as lua userdata object
        rc_Ptr<cnt_Params> d_params;

    };

    /** lua interface functions for scr_MCparm object.
     *
     * One way to use it:
     *
     * Initialize lua namespace (script_MCparm)
     * ~~~{.lua}
     require('milde')
     mc = require('libmclua')
     * ~~~
     * - At this point have <TT>libmclua:MINIBATCH_SGD</TT> (etc) enum constants
     *   - and a \c libmclua.mc (table) containing    'new' --> 'function'
     *   - and have set \c mc to the result of invoking \c libmclua.mc.new()
     * - To selectively use the debug libmclua (hopefully with debug libmcfilter)
     *   - create a link mcparm.so --> libmclua-dbg.so OR libmclua.so
     *   - and use \c mc=require('mcparm')
     * Construct default parameters (+ print, convert to lua table)
     * ~~~{.lua}
     p = mc             -- default parms
     print(p:str())     -- print: empty, because all values at default
     print(p:str(true)) -- print, verbose option
     t=p.get()          -- lua table (empty) of nondefault parms
     t=p.get(true)      -- lua table of all parms
     * ~~~
     * Modify parameters
     * ~~~{.lua}
     p = mc.new()
     t={}
     t.no_projections = 7.7777        -- int, ignores trailing .77777
     t.avg_epoch = 10                 -- size_t
     t.update_type = "MINIBATCH_SGD"  -- enum, using full value
     t.eta_type = "sqrt"              -- enum, using a lowercase substr
     t.C1 = 7.7777                    -- double, so .7777 not ignored
     t.resume = 1                     -- bool
p:set(t) -- <------- existing keys of p are overwritten by any settings in t
print(p:str())    -- now have some nondefault parameters
     * ~~~
     *
     * Note: getargs and setargs use the normal milde version, based on \c Args
     *       but I prefer the get/set versions based on \c ArgMap.
     *
     * - Alternate usage
     *   - maybe nicer, if want to extend \em mc with 'solve' function
     *     - i.e. more than just script_MCparm can fit into it :)
     * ~~~{.lua}
     require("milde")
     require("libmclua")
     -- create a lua mcparm object
     mc = libmclua.mc.new()     -- the only function in namespace libmclua.mcparm
     print(mc:str(true))
     print(mc:type())
     print(mc:str())
     print(mc:str(true))                           -- verbose listing (a handy reference)
     tab=mc:getargs()                              -- get all NON-DEFAULT args as string table
     for k,v in pairs(tab) do print(k,v) end       -- this will be empty
     tabargs=mc:getargs(true)                      -- get ALL args as table:str->str
     for k,v in pairs(tabargs) do print(k,v) end   -- now non-empty
     tab=mc:getargs(true)                          -- get ALL args as table:str->lua_type
     for k,v in pairs(tab) do print(k,v) end       -- ("looks" the same as printout of tabargs)
     -- note: enums are not converted to string nicely?
     * ~~~
     * Consider:
     * - mc.load( [fname:str] )                                 -- attach x,y data from repo-like format
     *   - just try various combos of slc/mlc and dr4/sr4, but \em not as milde repo
     *   - read into \b Eigen types (currently DenseM/SparseM of \em double )
     *     - because milde repo handles row-wise data, and \em MCFilter requires matrix x,y input
     * - mc.solve( [("fname.soln"):str], [ascii(false):bool] )  -- solve and write MCsoln
     */
    struct script_MCparm {
        static int f___gc();
        static int f_type();    ///< mcparm
        static int f_str();     ///< str([verbose:bool=0]) nondefault (or all) parms, one per line
        static int f_pretty();  ///< return 'pretty' string of the parameters
        static int f_help();    ///< return 'help' string for set({argstring}) command-line parameters
        // f_get may be a bit more convenient to use than getargs
        //    E.g., don't need tonumber(string) for assertions.
        static int f_get();     ///< string -> lua builtin type -- copy of MCfilter parms (ArgMap)
        static int f_getargs(); ///< string -> string table -- copy of MCfilter parms (Args)
        /** overwrite parameter entries set({table}) or set({argstring}) */
        static int f_set();
        static int f_setargs(); ///< set(<table>) (string keys) overwrite parms

        // constructors
        static int f_new();  ///< returns a [lua] default-constructed "param_struct"

        // utility functions:
        /** convert a \em lua slc/mlc dense/sparse repo to "plain Eigen" .x and .y files.
         * - Function:
         *   - produce Eigen-compatible data files compatible with libmcfilter
         *     'solve' and 'project' operations
         * - Lua Usage:
         *   - <B>repo2xy( repo, basename )</B>
         *     - output basename.x and basename.y files compatible with reading in
         *       as Eigen DenseM and SparseMb types
         *     - basename.x written always as float, with <B>D</B>ense/<B>S</B>parse as per repo
         *       - readable into one of <B>D</B>enseM or <B>S</B>parseM
         *       - header magic "MCx<B>D</B>" or "MCx<B>S</B>"
         *     - basename.y written <em>as 0..nClass numeric labels</em>,
         *       - readable into a SparseMb (sparse matrix of bool)
         *       - header magic "MCyN" (perhaps always written as bitmap??)
         * - except for header magics, \em similar to code written for
         *   {mcgen|mcsolve|mcproj}.cpp standalone (all-C++) utilities
         */
        static int f_toxy();

        // MCsolver/MCsoln functions
        //static int f_load();    ///< load(<fname:str>) -> 0/1 ~ bad/good
        static int f_xfile();   ///< xfile(<fname:str>) -- fname.x example Eigen data
        static int f_yfile();   ///< xfile(<fname:str>) -- fname.x example Eigen data
        static int f_solnfile();   ///< xfile(<fname:str>) -- fname.x example Eigen data
    };

#if 0
            // one option is for solve/project to build up program arguments with little lua calls
            //std::string xfile;
            //std::string yfile;
            //std::string solnfile;
            //std::string solnfmt;    ///< [default "SB"] 2 chars, S|L B|T for Short/Long Binary/Text soln format
            // threads?
            // xnorm?       NO!
            //std::string outputfile; ///< mcsolve operation may write a \em different solnfile as output.
            // possibly read these now, could also leave for later (during 'solve' or 'project' ops)
            //DenseM x;
            //bool isDense;
            //SparseM y;
            // simpler option is just to take all args as a string of command line parameters,
            // and whose boost::program_options mirrors the standalone utilites
            //    mcsolve,   mcproj,    and [perhaps even] mcgen
#if 0 // *** NEW ***
            std::string solveArgs;
            std::string projArgs;
#endif
#if 1 // XXX probably should be in a separate 'solver' object
        static std::string help_solve();        ///< return mcsolve --help string
        void solve( std::string solveArgs );    ///< solution "just like" standalone mcsolve
#endif
#if 0
        static std::string help_proj();         ///< return mcproj --help string
        static void proj( std::string projArgs );
#endif
#endif
    /** internal C++ impl for MCsolveProgram MCfilter function */
    struct scr_MCsolve{
        /** construct solver as lua userdata object.
         * Note that \c MCsolveArgs is a super-set of \c param_struct,
         * also allowing <em>solve</em>-specific settings (xfile,yfile,solnfile,...).
         */
        static scr_MCsolve* new_stack( int argc, char**argv, param_struct const* const defparms=nullptr );

        /** a ref-counted wrapper for a libmcfilter param_struct */
        struct cnt_Solve :
            //public Base_tag,   // defined in milde lib/os/util_io.hpp
            public rc_Cnt,
            public ::opt::MCsolveProgram
        {
            typedef ::opt::MCsolveProgram base;
            cnt_Solve( int argc, char** argv, param_struct const* const defparms )
                : ::opt::MCsolveProgram( argc, argv, /*verbose=*/1, defparms )
            {}
        };//class cnt_Solve

        virtual ~scr_MCsolve() {}

        rc_Ptr<cnt_Solve> d_solve;

    private:
        explicit scr_MCsolve(int argc, char**argv, param_struct const* const defparms=nullptr)
            : d_solve(new cnt_Solve(argc,argv,defparms))
        {}
    };
    /** lua interface functions for scr_MCsolve object.
     * - Actually, should create a solver class that, besides \c solve_optimization,
     *   provides write|read_binary|ascii of internal state.
     * - Internal state should be stored in short|full format
     *   - first store time t and time t state like eta, C1, C2, ...
     *   - then store
     *     - either short restart data from time-averaged {w,l,u,objective}
     *     - or long restart data that adds {w,l,u,obj} at time "t"
     */
    struct script_MCsolve {
        static int s___gc();
        static int s_type();            ///< mcsolve
        static int s_help();            ///< return help string for MCsolveProgram cmdline args
        //static int f_cmdline();         ///< return full constructor string (for later runs)
        /// \name s_FOO() --> MCsolveProgram::tryFOO()
        //@{
        static int s_read();
        static int s_solve();
        static int s_save();
        static int s_display();
        //@}

        /** constructor.
         * - 2 lua call types:
         *   - lua: new("--eta0=0.1 --etamin=1.e-3")
         *     - override <em>default settings</em> via cmdline args string
         *     - Note: if --solnfile, then <em>default settings</em> come from
         *             the .soln file (to nicely continue an existing run)
         *   - or   new(<mcparm>, "--etatype=SQRT --optlu=700 --maxiter=500 ...")
         *     - override user-specified defaults via cmdline args string
         */
        static int s_new();
    private:
        //struct param_struct pInit;
        //std::string cmdLine;
    };

}//MILDE::
#endif // MCLUA_H
