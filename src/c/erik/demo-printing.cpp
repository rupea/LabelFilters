
#include "printing.hh"

#include <assert.h>
#include <fstream>

using namespace std;
using namespace detail;         // io_txt, io_bin

int main(int,char**)
{
    typedef struct {
        uint32_t u;
        float    f;
        double   d;
        bool     b;
        string   s;
    } Data;
    Data data = { 111U, 222.222f, 333.333, true, string("Hello World") };
    {
        cout<<"io_txt test to cout..."<<endl;
        io_txt( cout, data.u );
        io_txt( cout, data.f );
        io_txt( cout, data.d );
        io_txt( cout, data.b );    // non-portable sizeof(bool) may vary
        io_txt( cout, data.s );
    }
    {
        cout<<"io_txt write to ofstream file test.txt"<<endl;
        ofstream f("test.txt");
        io_txt( f, data.u );
        io_txt( f, data.f );
        io_txt( f, data.d );
        io_txt( f, data.b );    // non-portable sizeof(bool) may vary
        io_txt( f, data.s );
        f.close();
    }
    {
        Data txt = { 0U, 0.0f, 0.0, false, string() };
        cout<<"io_txt read from ifstream file test.txt"<<endl;
        ifstream f("test.txt");
        io_txt( f, txt.u ); cout<<"txt.u = "<<txt.u<<endl; assert(txt.u == data.u);
        io_txt( f, txt.f ); cout<<"txt.f = "<<txt.f<<endl; assert(txt.f == data.f);
        io_txt( f, txt.d ); cout<<"txt.d = "<<txt.d<<endl; assert(txt.d == data.d);
        io_txt( f, txt.b ); cout<<"txt.b = "<<txt.b<<endl; assert(txt.b == data.b);
        io_txt( f, txt.s ); cout<<"txt.s = "<<txt.s<<endl; assert(txt.s == data.s);
        cout<<"Excellent -- txt write/read of some basic types worked"<<endl;
        assert( !f.eof() ); // SURPRISE - we don't yet "know" there is nothing more

        char unreadable = 'E';
        io_txt( f, unreadable );
        assert( f.fail() );
        assert( f.eof() );
        assert( unreadable == 'E' );
        f.close();
    }
    {
        cout<<"io_bin write to ofstream file test.bin"<<endl;
        ofstream f("test.bin");
        io_bin( f, data.u );
        io_bin( f, data.f );
        io_bin( f, data.d );
        io_bin( f, data.b );    // non-portable sizeof(bool) may vary
        io_bin( f, data.s );
        f.close();
    }
    {
        Data bin = { 0U, 0.0f, 0.0, false, string() };
        cout<<"io_bin read from ifstream file test.bin"<<endl;
        ifstream f("test.bin");
        io_bin( f, bin.u ); cout<<"bin.u = "<<bin.u<<endl; assert(bin.u == data.u);
        io_bin( f, bin.f ); cout<<"bin.f = "<<bin.f<<endl; assert(bin.f == data.f);
        io_bin( f, bin.d ); cout<<"bin.d = "<<bin.d<<endl; assert(bin.d == data.d);
        io_bin( f, bin.b ); cout<<"bin.b = "<<bin.b<<endl; assert(bin.b == data.b);
        io_bin( f, bin.s ); cout<<"bin.s = "<<bin.s<<endl; assert(bin.s == data.s);
        cout<<"Excellent -- bin write/read of some basic types worked"<<endl;
        assert( !f.eof() ); // SURPRISE - we don't yet "know" there is nothing more

        char unreadable = 'E';
        io_bin( f, unreadable );
        assert( f.fail() );
        assert( f.eof() );
        assert( unreadable == 'E' );
        f.close();
    }

    cout<<"\nGoodbye"<<endl;
}
