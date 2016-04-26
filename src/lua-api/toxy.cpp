
#include "mclua.hpp"

#include "base/app_state.hpp"
#include "lua.hpp"
#include "script_lua/lua_interpreter.hpp"

#include "printing.hh"

namespace MILDE{
    int scr_MCparm::toxy( scr_Repo const& r, std::string basename ){
#if 0
        // follow code of mcgen or printing.hh to write plain eigen row-wise matrix data
        // write rows, cols
        char const* const data_typstr = r.data_typstr();
        if( data_typstr()[0] == 'd' ){
            // dense data
            // Why is it SO difficult to convert from repo data to Matrix ?
            for(size_t i=0U; i<r.rows(); ++i){
                if( r.row(i) != 1 ) throw std::runtime_error("failed to access repo row");
                eigen_io_bin( output binary )
                    lua_pop(1);     // or something
            }else{
                // sparse data
            }
#elif 0
            int status;
            scr_TRY( "scr_MCparm::toxy(repo,basename)" ){
                status = r.rows();
                assert( status == 1 );
                scr_INT( 

                        status = r.cols();
                        assert( status == 1 );
                        if( r.data_typstr() == "dr4" ){
                        assert(true);
                        }
#endif
        throw std::runtime_error("scr_MCparm::toxy Not Implemented");
    }

}//MILDE::
