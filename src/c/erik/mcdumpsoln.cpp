
#include "find_w.h"
#include "printing.hh"
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <errno.h>      // errno
#include <string.h>     // strerror
#include <ctype.h>      // tolower

using namespace std;

// program options
string ifname;
string ofname;
bool binary=false;
bool pretty=false;
bool quiet=false;

// parse program options
void options(int argc, char**argv) {
    ifname.assign("");
    ofname.assign("");
    binary = false;
    pretty = false;
    quiet = false;
    bool dohelp=false;
    ostringstream oss;
    for(int a=1; a<argc; ++a){
        for(char* ca=argv[a]; *ca!='\0'; ){
            if(*ca == '-')                      ;/*ignore it*/
            else if(tolower(*ca) == 'i')        {ifname.assign(++ca); break;}
            else if(tolower(*ca) == 'b')        {binary=true ; ofname.assign(++ca); break;}
            else if(tolower(*ca) == 't')        {binary=false; ofname.assign(++ca); break;}
            else if(tolower(*ca) == 'p')        pretty=true;
            else if(tolower(*ca) == 'q')        quiet=true;
            else{
                if(tolower(*ca) != 'h'){
                    oss<<" unrecognized option character: "<<*ca<<endl;
                }
                dohelp=true;
            }
            ++ca;
        }
    }
    if(!quiet && !dohelp){
        cerr<<"mcdumpsoln input "<<(ifname.size()? ifname: string("cin"));
        if( pretty ){
            cerr<<(pretty? " pretty->cerr":"");
            if( ofname.size() == 0U ) cerr<<" NO .soln output";
        }
        if( !pretty || ofname.size() ){
            cerr<<(binary? ", text":", binary")<<" output to "<<(ofname.size()? ofname: string("cout"));
        }
        cerr<<endl;
    }
    if(dohelp){
        cerr<<" mcdumpsoln [Options]"
            "\n   This exercises MCsoln::read and MCsoln::write calls"
            "\n"
            "\nOptions: option character '-' is ignored, unrecognized --> this help"
            "\n         option characters, itbph, are case-insensitive"
            "\n   i<ifname>      INPUT filename        (no option | ifname absent => cin)"
            "\n   {t|b}[ofname]  [text]/binary OUTPUT  (no option | ofname absent => cout)"
            "\n   p              pretty-print to cerr, and skip any output to cout"
            "\n   q              quieter, skip a few informative messages to cerr"
            "\n   h              this help"
            "\n"
            "\nExamples:"
            "\n 0. [opt.] run mcgenx to generate some small .soln files"
            "\n 1. convert any .soln to short format text (to file, or cout"
            "\n      cat mcgen-a3-bin.soln | mcdumpsoln > my-txt.soln"
            "\n      mcdumpsoln -imcgen-a3-txt.soln     # to cout"
            "\n      mcdumpsoln -Imcgen-a3-bin.soln -Tmy-txt.soln"
            "\n 2. prettier text view of any .soln file"
            "\n      mcdumpsoln P < mcgen-a3-txt.soln "
            "\n      cat mcgen-a3-bin.soln | mcdumpsoln p 2| less"
            "\n      mcdumpsoln -QPImcgen-a3-bin.soln"
            "\n 3. rewrite any .soln as short binary"
            "\n      cat mcgen-a3-bin.soln | mcdumpsoln -b > bmy-bin.soln"
            "\n      mcdumpsoln imcgen-a3-txt.soln bmy-bin.soln"
            "\n      mcdumpsoln Imcgen-a3-txt.soln QBmy-bin.soln # q: skip cerr msg"
            "\n 4. rewrite any .soln as short binary, and ALSO pretty-print"
            "\n      mcdumpsoln -Imcgen-a3-txt.soln -pbmy-bin.soln"
            "\n      cat mcgen-a3-txt.soln | mcdumpsoln PBmy-bin.soln"
            "\n    No!   cat mcgen-a3-txt.soln | mcdumpsoln pb > bmy-bin.soln 2|less"
            "\n    Why?  p with no ofname => my-bin.soln is NOT generated"
            "\n"
            "\n TODO: option for long format output (if present in input)?"
            <<oss.str()         // any unrecognized options ?
            <<endl;
        exit(0);
    }
}

size_t filesize(std::string fname){
    struct stat st;
    errno = 0;
    if( stat(fname.c_str(), &st) < 0 ){
        throw std::runtime_error(strerror(errno));
    }
    return static_cast<size_t>(st.st_size);  // size in bytes, (from off_t = long int)
}

int main(int argc,char**argv){
    options(argc,argv);
    MCsoln soln;
    if( ifname.size() ){
        ifstream ifs;
        try {
            if(!quiet){cerr<<"### Step 1: read "<<ifname<<" --> MCsoln"; cerr.flush();}
            size_t fsz = filesize( ifname );
            if( fsz == 0 ) throw std::runtime_error("Empty input file");
            if(!quiet){cerr<<" ... input file size "<<fsz<<" bytes"<<endl;}
            ifs.open( ifname );
            if(!ifs.good()) throw std::runtime_error("Trouble opening input file");
            soln.read(ifs);
            ifs.close();
        }catch(std::exception const& what){
            if(!quiet){cerr<<"### OHOH! Error during read of MCsoln from "<<ifname<<endl;}
            ifs.close();
            throw(what);
        }
    }else{
        try {
            if(!quiet){cerr<<"### Step 1: read cin --> MCsoln"<<endl;}
            soln.read(cin);
        }catch(std::exception const& what){
            if(!quiet){cerr<<"### OHOH! Error during read of MCsoln from cin"<<endl;}
            throw(what);
        }
    }
    if(pretty){
        soln.pretty(cerr); // kinda pretty-print
    }
    if( ofname.size() ){
        ofstream ofs;
        try {
            if(!quiet){cerr<<"### Step 2: write MCsoln --> "<<ofname<<endl;}
            ofs.open(ofname);
            if(!ofs.good()) throw std::runtime_error("Trouble opening output file");
            soln.write( ofs, (binary? MCsoln::BINARY: MCsoln::TEXT), MCsoln::SHORT );
            ofs.close();
        }catch(std::exception const& what){
            if(!quiet){cerr<<"OHOH! Error during write of "<<(binary? "binary":"text")<<" .soln to "<<ofs<<endl;}
            ofs.close();
            throw(what);
        }
        if(!quiet){cerr<<" Good. "<<ofname<<" has "<<filesize(ofname)<<" bytes"<<endl;}
    }else if( !pretty ){        // -p and NO output spec means JUST prettyprint
        try {
            if(!quiet){cerr<<"### Step 2: write MCsoln --> cout as "<<(binary? "binary":"text")<<endl;}
            soln.write( cout, (binary? MCsoln::BINARY: MCsoln::TEXT), MCsoln::SHORT );
        }catch(std::exception const& what){
            if(!quiet){cerr<<"\n\nOHOH! Error during write of "<<(binary? "binary":"text")<<" .soln to cout"<<endl;}
            throw(what);
        }
    }
    if(!quiet){cerr<<"\nGoodbye"<<endl;}
}
