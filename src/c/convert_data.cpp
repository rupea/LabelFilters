#include "mcxydata.h"
#include <boost/program_options.hpp>
#include <iostream>

using namespace std;
namespace po=boost::program_options;
using namespace po;

int main(int argc, char**argv){

  po::variables_map vm;

  po::options_description desc;      
  try {
    
    desc.add_options()
      ("dense,d",value<bool>()->default_value(false)->implicit_value(true),"Convert to dense format. Default is sparse.")
      ("output,o", value<std::string>(), "Output file")
      ("help","This help")
      ;
    po::options_description all;      
    all.add(desc).add_options()
      ("input,i",value<string>()->required(),"Input file"); // hidden option

    po::positional_options_description posopt;
    posopt.add("input", 1);
    
    po::store(po::command_line_parser(argc, argv).options(all)
	      .positional(posopt).run(),
	      vm); 

    if( vm.count("help") ) {
      cout << endl << "Usage: " << argv[0] << " [options] input_file" << endl<<endl;
      cout<<desc<<endl;
      exit(0);
    }

    po::notify(vm); // at this point, raise any exceptions for 'required' args
    
  }catch(std::exception const& e){
    cerr << "Error: "<<e.what()<<endl<<endl<<endl;
    cerr << "Usage: " << argv[0] << " [options] input_file" << endl << endl;
    cerr << desc << endl;
    exit(-1);
  }catch(...){
    cerr<<"Command-line parsing exception of unknown type!"<<endl<<endl;
    cerr << "Usage: " << argv[0] << " [options] input_file" << endl <<endl;
    cerr << desc << endl;
    exit(-2);
  }


  string infile = vm["input"].as<string>();

  bool dense = vm["dense"].as<bool>();

  
  string outfile;
  if (vm.count("output"))
    {
      outfile = vm["output"].as<string>();
    }
  else
    {
      outfile = infile + (dense?".dense":".sparse") + (".bin");
    }

  
  MCxyData data;

  data.read(infile);

  if (dense)
    data.toDense();
     
  data.xwrite(outfile + ".X");
  data.ywrite(outfile + ".Y");

}
    
  
