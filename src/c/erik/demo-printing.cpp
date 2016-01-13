
#include "printing.hh"

#include <assert.h>
#include <fstream>
#include <array>
#include <sstream>

using namespace std;
using namespace detail;         // io_txt, io_bin
using namespace boost;

string bits3a="010";
string bits3b="101";
string bits73a="0010001011110000000100000000000000000000100000000000000000000000000000000";
string bits73b="1110000000000000000000010000001000000000000000000001110000000000000010001";
int main(int,char**)
{
    if(1){
        std::array<int,3> a={1,2,3};
        cout<<" std::array<int,3> -->"; for(auto& x: a) cout<<' '<<x; cout.flush();
        cout<<" io_txt-->"; cout.flush(); io_txt(cout,a," "); cout<<' '; cout.flush();
        std::array<int,3> b;
        stringstream ss;
        io_txt(static_cast<ostream&>(ss),a," ");        // ws=" " shortens printout

        cout<<" ss="<<ss.rdbuf(); cout.flush();
        ss.clear(); ss.seekg(0); //rewind ss

        io_txt(static_cast<istream&>(ss),b);
        for(uint32_t i=0U; i<b.size(); ++i) assert( b[i] == a[i] );
        cout<<" OK"<<endl;
    }
    if(1){
        std::array<char,3> a={'y','a','h'};
        cout<<" std::array<char,3> a = "; for(auto& x: a) cout<<' '<<x; cout.flush();
        cout<<" io_txt-->"; cout.flush(); io_txt(cout,a); cout<<' '; cout.flush();
        std::array<char,3> b;
        stringstream ss;
        io_txt(static_cast<ostream&>(ss),a," ");

        cout<<" ss.str()="<<ss.str(); cout.flush();
        //cout<<" ss.rdbuf()="<<ss.rdbuf(); cout.flush(); // not nice, since REQUIRE rewind to re-use
        //ss.clear(); ss.seekg(0); //rewind ss, REQUIRED after rdbuf has been emptied

        io_txt(static_cast<istream&>(ss),b);
        //cout<<" ... b  = "; for(auto& x: b) cout<<' '<<x; cout.flush();
        cout<<" ... b  = "; for(uint32_t i=0U; i<b.size(); ++i) cout<<" b["<<i<<"]="<<b[i]; cout<<" "; cout.flush();
        for(uint32_t i=0U; i<3U; ++i) assert( b[i] == a[i] );

        //ss.clear(); ss.seekg(0);                // rewind
        cout<<" rdbuf --> "<<ss.str();  // still full-length -- rewind not necessary for str()
        cout<<" rdbuf --> "<<ss.str();  // still full-length
        cout<<" OK"<<endl;
    }
    if(1){
        dynamic_bitset<> a(5);
        a[0] = false;
        a[1] = false;
        a[2] = true;
        a[3] = true;
        a[4] = true;
        assert( a[0]==0 ); assert( a[1]==0 ); assert( a[2]==1 ); assert( a[3]==1 ); assert( a[4]==1 );
        cout<<"dynamic_bitset<> a = 00011(little-endian): TEXT I/O IS BIG_ENDIAN";
        cout<<" --> operator<< : "<<a<<' '; cout.flush();
        cout<<" io_txt : "; io_txt(cout,a," "); cout.flush();
        cout<<" to_string : "; string s; to_string(a,s); cout<<s; cout.flush();
        cout<<endl;
        dynamic_bitset<> b;
        stringstream ss;
        io_txt(static_cast<ostream&>(ss),a);
        io_txt(static_cast<istream&>(ss),b);
        for(uint32_t i=0U; i<5U; ++i) assert( b[i] == a[i] );
        cout<<" OK1"; cout.flush();
        // try again, with same stringstream
        dynamic_bitset<> c;
        ss.clear(); ss.seekg(0);                        // without "rewind" --> segfault
        io_txt(static_cast<istream&>(ss),c); 
        for(uint32_t i=0U; i<5U; ++i) assert( c[i] == a[i] );
        cout<<" OK"<<endl;
    }
    if(1){
        dynamic_bitset<> bs(bits73b);
        stringstream iss(" \n\t 100111\n  12"); // NOTE will be read in as BIG-ENDIAN
        iss>>bs;
        //for(uint32_t i=0U; i<bs.size(); ++i) cout<<" bs["<<i<<"] = "<<bs[i]<<endl; // confirm big-endian
        cout<<"bitset: size="<<bs.size()<<" nblocks="<<bs.num_blocks()<<" text "<<bs;
        cout<<" io_txt "; io_txt(cout,bs); 
        assert( bs.size() == 6 );
        assert( bs[0] == 1 );
        assert( bs[1] == 1 );
        assert( bs[2] == 1 );
        assert( bs[3] == 0 );
        assert( bs[4] == 0 );
        assert( bs[5] == 1 );
        cout<<" OK"<<endl;
    }
    if(1){
        dynamic_bitset<> a(bits73b);
        cout<<" boost::dynamic_bitset<> a(bits73) -->"; for(uint32_t i=0U; i<a.size(); ++i) cout<<a[i]; cout.flush();
        dynamic_bitset<> b;
        stringstream ss;
        io_txt(static_cast<ostream&>(ss),a);
        io_txt(static_cast<istream&>(ss),b);
        for(uint32_t i=0U; i<b.size(); ++i) assert( b[i] == a[i] );
        cout<<" OK"<<endl;
    }
    if(1){
        DenseM m;
        m.setRandom(2,3);
        cout<<" a random matrix... ---> DenseM.bin"<<endl;
        cout<<m<<endl;
        stringstream ss;
        ss<<m;                  // OK
        ofstream ofs("DenseM.bin");
        eigen_io_bin(ofs,m);
        ofs.close();
    }
    if(1){
        cout<<"reading from DenseM.bin...";
        DenseM n(2,3);
        n.setZero();
        {
            ifstream ifs; ifs.open("DenseM.bin");
            eigen_io_bin(ifs,n);
            ifs.close();
        }
        cout<<" done reading DenseM.bin ";
        cout<<" read bin --->\n"<<n<<endl;
        //ss>>n;                  // Eigen does NOT provide this
        // error message is confusing:  cannot bind ‘std::basic_istream<char>’ lvalue to ...&&
        //cout<<n<<endl;
    }
    if(1){
        DenseM m;
        m.setRandom(2,3);
        cout<<"\na random matrix... ---> DenseM.txt"<<endl;
        cout<<m<<endl;  // NOT enough -- it does not output the dimensions
        eigen_io_txt(cout,m);
        stringstream ss;
        ss<<m;                  // OK
        {
            ofstream ofs("DenseM.txt");
            eigen_io_txt(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from DenseM.txt...";
        DenseM n(2,3);
        n.setZero();
        //cout<<" orig n(2,3) --->\n"<<n<<endl;
        {
            ifstream ifs("DenseM.txt");
            eigen_io_txt(ifs,n);
            ifs.close();
        }
        cout<<" done reading DenseM.txt ";
        cout<<" read txt --->\n"<<n<<endl;
        //ss>>n;                  // Eigen does NOT provide this
        // error message is confusing:  cannot bind ‘std::basic_istream<char>’ lvalue to ...&&
        //cout<<n<<endl;
        assert(n.rows() == m.rows());
        assert(n.cols() == m.cols());
        for(uint32_t r=0U; r<m.rows(); ++r)
            for(uint32_t c=0U; c<m.cols(); ++c)
                assert( fabs(m(r,c)-n(r,c))<1.e-6 );
    }
    if(1){
        VectorXd m;
        m.setRandom(3);
        cout<<"\na random vector... ---> VectorXd.txt "<<endl;
        cout<<m.transpose()<<endl;  // NOT enough -- it does not output the dimensions
        eigen_io_txt(cout,m);
        stringstream ss;
        ss<<m;                  // OK
        {
            ofstream ofs("VectorXd.txt");
            eigen_io_txt(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from VectorXd.txt...";
        VectorXd n(3);
        n.setZero();
        //cout<<" orig n(3).transpose() ---> "<<n.transpose()<<endl;
        {
            ifstream ifs("VectorXd.txt");
            eigen_io_txt(ifs,n);
            ifs.close();
        }
        cout<<" done reading VectorXd.txt";
        cout<<" ---> n.transpose =\n"<<n.transpose()<<endl;
        assert(n.size() == m.size());
        for(uint32_t i=0U; i<m.size(); ++i) assert( fabs(m[i]-n[i])<1.e-6 );
    }
    if(1){
        VectorXd m;
        m.setRandom(3);
        cout<<"\na random vector... ---> VectorXd.bin, ";
        cout<<"m.transpose = "<<m.transpose()<<endl;  // NOT enough -- no dims output
        {
            ofstream ofs("VectorXd.bin");
            eigen_io_bin(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from VectorXd.bin...";
        VectorXd n(2);
        n.setZero();
        //cout<<" orig n(2).transpose() ---> "<<n.transpose()<<endl;
        {
            ifstream ifs("VectorXd.bin");
            eigen_io_bin(ifs,n);
            ifs.close();
        }
        cout<<" done reading VectorXd.bin ";
        cout<<" read bin ---> n.transpose = "<<n.transpose()<<endl;
        assert(n.size() == m.size());
        for(uint32_t i=0U; i<m.size(); ++i) assert( fabs(m[i]-n[i])<1.e-6 );
    }
    if(1){
        boolmatrix m(22,33);
        m.set(2,3); m.set(2,4);
        ofstream ofs("boolmatrix.bin");
        io_bin(ofs,m);
        ofs.close();
        ifstream ifs("boolmatrix.bin");
        boolmatrix n(22,33);    // boolmatrix dims never change
        io_bin(ifs,n);
        ifs.close();
        assert( n.cbase() == m.cbase() );       // cbase returns the const boost::dynamic_bitset "base"
        cout<<" boolmatrix.bin --- OK"<<endl;
    }
    if(1){
        cout<<"\na sparse matrix... ---> SparseM.txt"<<endl;
        int const cols=4;
        SparseM m(2,cols);                         // ROW-major
        m.reserve(VectorXi::Constant(cols,3));     // reserve mem for 2 nz per ROW
        m.insert(0,1)=0.1;
        m.insert(0,3)=0.3;
        m.insert(1,0)=1.0;
        m.insert(1,2)=1.2;
        if(1){ // visualize the text format, should be identical for compressed or not
            //cout<<m<<endl;  // no matching function, or assertion
            eigen_io_txt(cout,m);
            m.makeCompressed();     // different way to output...
            eigen_io_txt(cout,m);
        }
        if(0){ // Eigen operator<< is quite verbose, skip
            stringstream ss;
            ss<<m;                  // OK
            cout<<" ss.str() --> "<<ss.str()<<endl;
        }
        {
            ofstream ofs("SparseM.txt");
            eigen_io_txt(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from SparseM.txt..."; cout.flush();
        SparseM n(2,3);
        n.setZero();
        //cout<<" orig n(2,3) --->\n"<<n<<endl;
        {
            ifstream ifs("SparseM.txt");
            eigen_io_txt(ifs,n);
            ifs.close();
        }
        cout<<" done reading SparseM.txt ";
        //cout<<" read txt --->\n"<<n<<endl;
        //ss>>n;                  // Eigen does NOT provide this
        // error message is confusing:  cannot bind ‘std::basic_istream<char>’ lvalue to ...&&
        //cout<<n<<endl;
        assert(n.rows() == m.rows());
        assert(n.cols() == m.cols());
        for(int r=0U; r<m.rows(); ++r)
            for(int c=0U; c<m.cols(); ++c)
                assert( fabs(m.coeff(r,c)-n.coeff(r,c))<1.e-6 );
        cout<<" SparseM.txt read back identical -- OK"<<endl;
    }
    if(1){
        stringstream ss;
        ostream& os=ss;
        io_bin(os,1.11f);
        io_bin(os,2.22);
        io_bin(os,3.33f);
        istream& is=ss;
        float f1; io_bin(is,f1);
        double d; io_bin(is,d);
        float f2; io_bin(is,f2);
        assert( fabs(f1-1.11f)<1.e-6 );
        assert( fabs(d -2.22 )<1.e-6 );
        assert( fabs(f2-3.33f)<1.e-6 );
    }

    if(1){
        cout<<"\na sparse uncompressed matrix... ---> SparseM.bin"<<endl;
        int const cols=4;
        SparseM m(2,cols);                         // ROW-major
        m.reserve(VectorXi::Constant(cols,3));     // reserve mem for 2 nz per ROW
        m.insert(0,1)=0.1;
        m.insert(0,3)=0.3;
        m.insert(1,0)=1.0;
        m.insert(1,2)=1.2;
        {
            ofstream ofs("SparseM.bin");
            eigen_io_bin(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from SparseM.bin..."; cout.flush();
        SparseM n(13,13); // ignored
        n.setZero();
        //cout<<" orig n(2,3) --->\n"<<n<<endl;
        {
            ifstream ifs("SparseM.bin");
            eigen_io_bin(ifs,n);
            ifs.close();
        }
        cout<<" done reading SparseM.bin ";
        //cout<<" read bin --->\n"<<n<<endl;
        //ss>>n;                  // Eigen does NOT provide this
        // error message is confusing:  cannot bind ‘std::basic_istream<char>’ lvalue to ...&&
        //cout<<n<<endl;
        assert(n.rows() == m.rows());
        assert(n.cols() == m.cols());
        for(int r=0U; r<m.rows(); ++r)
            for(int c=0U; c<m.cols(); ++c)
                assert( fabs(m.coeff(r,c)-n.coeff(r,c))<1.e-6 );
        cout<<" SparseM.bin read back identical -- OK"<<endl;
    }
    if(1){
        cout<<"\na sparse compressed matrix... ---> SparseM.bin"<<endl;
        int const cols=4;
        SparseM m(2,cols);                         // ROW-major
        m.reserve(VectorXi::Constant(cols,3));     // reserve mem for 2 nz per ROW
        m.insert(0,1)=0.1;
        m.insert(0,3)=0.3;
        m.insert(1,0)=1.0;
        m.insert(1,2)=1.2;
        m.makeCompressed();     // different way to output...
        assert( m.isCompressed() );
        {
            ofstream ofs("SparseM.bin");
            eigen_io_bin(ofs,m);  // WRONG: does not save size
            ofs.close();
        }
        cout<<"reading from SparseM.bin..."; cout.flush();
        SparseM n(13,13); // ignored
        n.setZero();
        //cout<<" orig n(2,3) --->\n"<<n<<endl;
        {
            ifstream ifs("SparseM.bin");
            eigen_io_bin(ifs,n);
            ifs.close();
        }
        cout<<" done reading SparseM.bin ";
        //cout<<" read bin --->\n"<<n<<endl;
        //ss>>n;                  // Eigen does NOT provide this
        // error message is confusing:  cannot bind ‘std::basic_istream<char>’ lvalue to ...&&
        //cout<<n<<endl;
        assert(n.rows() == m.rows());
        assert(n.cols() == m.cols());
        for(int r=0U; r<m.rows(); ++r)
            for(int c=0U; c<m.cols(); ++c)
                assert( fabs(m.coeff(r,c)-n.coeff(r,c))<1.e-6 );
        cout<<" SparseM.bin read back identical -- OK"<<endl;
    }




    typedef struct {
        uint32_t u;
        float    f;
        double   d;
        bool     b;
        string   s;
        dynamic_bitset<> bs1;
        dynamic_bitset<> bs2;
        string   s2;
    } Data;
    Data data = { 111U, 222.222f, 333.333, true, string("Hello World"), dynamic_bitset<>(bits3a), dynamic_bitset<>(bits73a), string("all done") };
    {
        cout<<"\n\nio_txt test to cout..."<<endl;
        io_txt( cout, data.u );
        io_txt( cout, data.f );
        io_txt( cout, data.d );
        io_txt( cout, data.b );    // non-portable sizeof(bool) may vary
        io_txt( cout, data.s );
        io_txt( cout, data.bs1 );
        io_txt( cout, data.bs2 );
        io_txt( cout, data.s2 );
    }
    {
        cout<<"\n\nio_txt write to ofstream file test.txt"<<endl;
        ofstream f("test.txt");
        io_txt( f, data.u );
        io_txt( f, data.f );
        io_txt( f, data.d );
        io_txt( f, data.b );    // non-portable sizeof(bool) may vary
        io_txt( f, data.s );
        io_txt( f, data.bs1 );
        io_txt( f, data.bs2 );
        io_txt( f, data.s2 );
        f.close();
    }
    {
        Data txt = { 0U, 0.0f, 0.0, false, string(), dynamic_bitset<>(bits3b), dynamic_bitset<>(), string("garbage") };
        cout<<"io_txt read from ifstream file test.txt"<<endl;
        //cout<<"\ttxt.bs1="<<txt.bs1<<"  data.bs1="<<data.bs1<<endl;
        ifstream f("test.txt");
        uint32_t nerr=0U;
#define TEST(PARM) io_txt( f, txt.PARM ); cout<<"txt." #PARM " = "<<txt.PARM; if(txt.PARM != data.PARM){ ++nerr; cout<<"OHOH, txt." #PARM " != data." #PARM;} cout<<endl; cout.flush();
        TEST(u);
        TEST(f);
        TEST(d);
        TEST(b);
        TEST(s);
        TEST(bs1);
        TEST(bs2);
        TEST(s2);
#undef TEST
        assert(nerr == 0U);
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
        cout<<"\nio_bin write to ofstream file test.bin"<<endl;
        ofstream f("test.bin");
        io_bin( f, data.u );
        io_bin( f, data.f );
        io_bin( f, data.d );
        io_bin( f, data.b );    // non-portable sizeof(bool) may vary
        io_bin( f, data.s );
        io_bin( f, data.bs1 );
        io_bin( f, data.bs2 );
        io_bin( f, data.s2 );
        f.close();
    }
    {
        Data bin = { 0U, 0.0f, 0.0, false, string(), dynamic_bitset<>(bits3b), dynamic_bitset<>(bits73b), string("garbage") };
        cout<<"io_bin read from ifstream file test.bin"<<endl;
        ifstream f("test.bin");
#define TEST(PARM) io_bin( f, bin.PARM ); cout<<"bin." #PARM " = "<<bin.PARM<<endl; assert(bin.PARM == data.PARM);
        TEST(u);
        TEST(f);
        TEST(d);
        TEST(b);
        TEST(s);
        TEST(bs1);
        TEST(bs2);
        TEST(s2);
#undef TEST
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
