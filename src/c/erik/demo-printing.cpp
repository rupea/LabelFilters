
#include "printing.hh"

#include <assert.h>
#include <fstream>

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
        dynamic_bitset<> bs(bits73b);
        stringstream iss(" \n\t 000111\n  12");
        iss>>bs;
        cout<<"bitset: size="<<bs.size()<<" nblocks="<<bs.num_blocks()<<" text "<<bs;
        cout<<endl;
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
