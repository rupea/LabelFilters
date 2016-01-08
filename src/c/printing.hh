#ifndef PRINTING_HH
#define PRINTING_HH

#include "printing.h"
#include <iostream>
#include <stdexcept>

namespace detail {
    template<typename T>
        inline std::ostream&
        io_txt( std::ostream& os, T const& x, char const* ws="\n" )
        { return os << x << ws; }
    template<typename T>
        inline std::istream&
        io_txt( std::istream& is, T& x )
        { return is >> x; }
    template<typename T>
        inline std::ostream&
        io_bin( std::ostream& os, T const& x )
        { return os.write(reinterpret_cast<char const*>(&x),sizeof(T)); }
    template<typename T>
        inline std::istream&
        io_bin( std::istream& is, T& x )
        { return is.read (reinterpret_cast<char*>(&x),sizeof(T)); }

    // specializations
    //   strings as length + (no intervening space) + blob
    template<> inline std::ostream& io_txt( std::ostream& os, std::string const& x, char const* ws/*="\n"*/ ){
        uint32_t len=(uint32_t)(x.size() * sizeof(std::string::traits_type::char_type));
        io_txt(os,len,"");      // no intervening whitespace
        if(os.fail()) throw std::overflow_error("failed string-len-->std::ostream");
        os<<x<<ws;
        if(os.fail()) throw std::overflow_error("failed string-data-->std::ostream");
        return os;
    }
    template<> inline std::istream& io_txt( std::istream& is, std::string& x ){
        uint32_t len;
        io_txt(is,len);
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        is.read(&x[0], len);    // read full string content
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-data");
        return is;
    }
    template<> inline std::ostream& io_bin( std::ostream& os, std::string const& x ){
        uint32_t len=(uint32_t)(x.size() * sizeof(std::string::traits_type::char_type));
        io_bin(os,len);
        if(os.fail()) throw std::overflow_error("failed string-len-->std::ostream");
        os.write(x.data(),len);
        if(os.fail()) throw std::overflow_error("failed string-data-->std::ostream");
        return os;
    }
    template<> inline std::istream& io_bin( std::istream& is, std::string& x ){
        uint32_t len;
        io_bin(is,len);
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        is.read(&x[0], len);    // read full string content
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-data");
        return is;
    }
}

template <typename Block, typename Alloc> inline
  void save_bitvector(std::ostream& out, const boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits = bs.size();
  size_t num_blocks = bs.num_blocks();
  std::vector<Block> blocks(num_blocks);
  to_block_range(bs, blocks.begin());  
  out.write((char*)&num_bits, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)&num_blocks, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)(&(blocks[0])), num_blocks*sizeof(Block));  
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
}

template <typename Block, typename Alloc> inline
  int load_bitvector(std::istream& in, boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits,num_blocks;
  in.read((char*)&num_bits, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  in.read((char*)&num_blocks, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  std::vector<Block> blocks(num_blocks);
  in.read((char*)(&(blocks[0])), num_blocks*sizeof(Block));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  bs.resize(num_bits);
  from_block_range(blocks.begin(), blocks.end(), bs);
  bs.resize(num_bits);
  return 0;
}

    template<typename EigenType> inline
void print_mat_size(const EigenType& mat)
{
    using namespace std;
    cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}
    template<> inline
void print_mat_size(const Eigen::VectorXd& mat)
{
    using namespace std;
    cout << "(" << mat.size() << ")";
}


    template<typename EigenType> inline
void print_report(const int projection_dim, const int batch_size,
                  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
                  const EigenType& x)
{
    using namespace std;
    cout << "projection_dim: " << projection_dim << ", batch_size: "
        << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
        << ", C2: " << C2 << ", lambda: " << lambda << ", size w: " << w_size << ", ";
    print_report(x);
    cout << "\n-----------------------------\n";

}

#endif // PRINTING_HH
