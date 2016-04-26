#include "mclua.hpp"
#include <assert.h>

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "script_lua/lua_interpreter.hpp"
#include "script_lua/lua_api.hpp"

#include <iostream>

#include "base/dynamic_object.hpp"   // $MILDE_DIR/src/lib/base/dynamic_object.hpp : init_MILDE_objfactory, ...
//#define init_MILDE_objfactory    if (!MILDE::GAS.d_objfactory) MILDE::GAS.d_objfactory = new MILDE::Map<std::string,MILDE::object_factory>;

using namespace std;
using namespace MILDE;

static void  __attribute__((unused)) l_message (const char *pname, const char *msg){
  if (pname) luai_writestringerror("%s: ", pname);
  luai_writestringerror("%s\n", msg);
}

/** error handler while calling milde functions directly.
 * When calling MiLDe function that may generate an error
 * they must be enclosed in try-catch. */
int __attribute((noreturn)) milde_err_handler(const MILDE_Exception& e)
{
  throw(e);
}

#define SCRAPIERR do { *mycout << ScrAPI_err << endl; retcode=-1; goto end; } while(0)
#define DBGEXEC( LUA_STUFF ) do { \
    std::cerr.flush(); std::cout.flush(); \
    std::string str; \
    { \
        std::ostringstream oss; \
        oss<<LUA_STUFF; \
        str = oss.str(); \
    } \
    cout<<std::setw(80)<<str<<" -- Executing..."<<std::endl; \
    if( !ScrAPI_execute_line(str.c_str()) ) { \
        { \
            str.insert(0,"Error: "); \
            std::ostringstream oss; \
            oss<<" : ScrAPI_err = "<<ScrAPI_err; \
            str.append(oss.str()); \
        } \
        /*SCRAPIERR*/ \
        throw( std::runtime_error( str )); \
    } \
    std::cout<<std::setw(80)<<str<<" -- OK"<<std::endl; \
} while(0)


int main(int argc,char**argv)
{
    if(0){ // init as in runmilde...
        // initialize MiLDe
        set_error_handler(milde_err_handler);
        init_MILDE_objfactory;
        // get directory where executable resides
        //string appdir = dirname(GAS.d_appname);
    }


#if 0
    // very low-level MIGHT be something like...
    lua_State *L = luaL_newstate();
    if( L == nullptr ){
        l_message(argv[0], " cannot create lua state: not enough memory");
        return EXIT_FAILURE;
    }
    /* open standard libraries */
    luaL_checkversion(L);
    lua_gc(L, LUA_GCSTOP, 0);  /* stop collector during initialization */
    luaL_openlibs(L);  /* open libraries */
    lua_gc(L, LUA_GCRESTART, 0);
#endif
    // But MILDE-ish startup can go something like ...
    ostream* mycout = &cout;
    int retcode = 0;
    try{
        // initialize MiLDe and mcparm  (mcparm.so linked to libmclua.so OR libmclua-dbg.so)
        // better tests:
        //      lua_cpp parms.lua
        set_error_handler(milde_err_handler);
        init_MILDE_objfactory;
        ScrAPI_init(mycout,mycout,mycout);
        DBGEXEC("require('milde')");
        DBGEXEC("mc=require('mcparm')");
        //script_MCparm x = script_MCparm();
        //scr_MCparm y = scr_MCparm();
        cout<<"\nGoodbye"<<endl;
    }catch(const MILDE_Exception& e){
        *mycout << e.txt() << endl;
        retcode= -2;
        goto end;
    }
end:
    ScrAPI_exit();
    return retcode;
}

#if 0
// ---------------- OLD ATTEMPTS -----------------------
#if 0
        // with above init, the following will get stuck in _dl_start_user...
        //                MILDE::milde_State::milde_State() constructor,
        //                waiting on omp_set_lock()
        attach_lua(nullptr);
        lua_Interpreter *li = MILDE_li();
        assert(li->d_lua != nullptr);
        cout<<" GAS.d_si = "<<GAS.d_si<<" , li = "<<li<<endl;
#endif
#if 0
        // initialize MiLDe [ONLY]
        set_error_handler(milde_err_handler);
        init_MILDE_objfactory;
        ScrAPI_init(mycout,mycout,mycout);
        if( !ScrAPI_execute_line("require('milde')") ) SCRAPIERR;
#elif 0
        //lua_Interpreter *li = MILDE_li();
        //li->ctor();
        attach_lua(nullptr);
        lua_Interpreter *li = MILDE_li();
        assert(li->d_lua != nullptr);
        cout<<" GAS.d_si = "<<GAS.d_si<<" , li = "<<li<<endl;
#elif 0
        luaopen_mcparm_base( li->d_lua );

        // the lua interpreter is not there, since GAS.d_si is still a nullptr.
        // ---> SIGSEGV as soon as f_type is called
        if( script_MCparm::f_new() != 1 ) throw(MILDE_Exception("script_MCparm","f_new failed"));
        li->put_user( "scr_MCparm", sizeof(scr_MCparm) );
#elif 0

        cout<<" y type is "<<script_MCparm::f_type()<<endl;
        //cout<<" y get_no_projections "<<script_MCparm::f_get_no_projections()<<endl;
#endif
#endif
