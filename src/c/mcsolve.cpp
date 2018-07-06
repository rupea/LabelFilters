/** \file
 * <B>M</B>ulti-<B>C</B>lass <B>solver</B> for discriminating projection lines.
 */

#include "mclearnFilter.h"
#include "mcxydata.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace boost::program_options;

int main(int argc, char**argv){
#ifndef NDEBUG
  int const verb = +1;        // verbosity modifier
#else
  int const verb = 0;         // verbosity modifier
#endif

  po::variables_map vm;
  int verbose= 1;
  
  try {
    po::options_description desc;
    po::options_description general("Genral Options");      
    
    general.add_options()      
      ("verbose,v",value<int>()->default_value(1),"Verbosity level")
#ifdef _OPENMP
      ("nthreads", value<int>()->default_value(0),"Number of threads to use. 0 means all available")
#endif
      ("outfile,o", value<string>()->default_value(""), "File to save the solution to")
      ("outbinary,b", value<bool>()->implicit_value(true)->default_value(false), "Save solution in binary format") 
      ("help,h","This help")
      ;
    
    desc.add(general).add(MCxyDataArgs::getDesc()).add(MCsolveArgs::getDesc());
    po::parsed_options parsed
      = po::command_line_parser( argc, argv )
      .options( desc )
      .run();
    po::store( parsed, vm );
    
    if( vm.count("help") ) {
      cout<<desc<<endl;
      exit(0);
    }
    
    po::notify(vm); // at this point, raise any exceptions for 'required' args
    
  }catch(po::error& e){
    cerr<<"Invalid argument: "<<e.what()<<endl;
    throw;
  }catch(std::exception const& e){
    cerr<<"Error: "<<e.what()<<endl;
    throw;
  }catch(...){
    cerr<<"Command-line parsing exception of unknown type!"<<endl;
    throw;
  }
  
  if (vm.count("verbose"))
    {
      verbose = vm["verbose"].as<int>() + verb;
    }


  string outfile;
  if (vm.count("outfile"))
    {
      outfile = vm["outfile"].as<string>();
    }

  bool outbinary = false;
  if (vm.count("outbinary"))
    {
      outbinary = vm["outbinary"].as<bool>();
    }
     
    
  MCxyDataArgs dataargs(vm);
  MCsolveArgs solverargs(vm);
  
  solverargs.params.verbose(verbose);
#ifdef _OPENMP
  if (vm.count("nthreads"))
    {
      int nthreads = vm["nthreads"].as<int>();
      solverargs.params.num_threads(nthreads);      
      if (nthreads > 0)
	{
	  omp_set_num_threads(nthreads);
	}
    }
#endif
  
  
  if (dataargs.xFiles.size() > 1)
    {
      cerr << "WARNING: label filters are leraned only on the first dataset. Ignoring the rest" << endl;
    }
  
  // get the stats for data normalization
  bool center = 0;
  std::string normfile;

  if (dataargs.normData.size() > 0)
    {
      normfile = dataargs.normData;
    }
  else
    {
      normfile = dataargs.xFiles[0]; // use the first file to calculate the stats for data normalization and removing rare features
    }

  Eigen::VectorXd mean;
  Eigen::VectorXd sdev;
  vector<size_t> feature_map;
  vector<size_t> reverse_feature_map;
  vector<size_t> label_map;
  vector<size_t> reverse_label_map;
  
  { // get statistics for normalization and for removal of rare features. 
    MCxyData data;
    
    if (dataargs.rmRareF > 0 || dataargs.rmRareL > 0 || dataargs.xnorm )
      {
	data.read(normfile);
      }
    if (dataargs.rmRareF > 0)
      {
	data.removeRareFeatures(feature_map, reverse_feature_map, dataargs.rmRareF, false);
      }
    
    if (dataargs.rmRareL > 0)
      {
	data.removeRareLabels(label_map, reverse_label_map, dataargs.rmRareL, false);
      }
    
    if (dataargs.xnorm)
      {
	switch (dataargs.center)
	  {
	  case -1: 				  		      
	    center = data.denseOk;
	    break;
	  case 0:
	    center = false;
	    break;
	  case 1:
	    center = true;
	    break;
	  default :
	    throw runtime_error("Unrecongnized argument center = "+to_string(dataargs.center));
	  }		
	data.xstdnormal(mean, sdev, true, center, false);
      }
  }
  
  // prepare the data
  string xfile = dataargs.xFiles[0];
  string yfile;
  if (dataargs.yFiles.size() == 0)
    {
      // no yfile
      yfile = "";
    }
  else
    {
      yfile = dataargs.yFiles[0];
    }

  shared_ptr<MCxyData> data = make_shared<MCxyData>();
  data->read(xfile, yfile);
  if (dataargs.rmRareF > 0)
    {
      data->removeRareFeatures(feature_map, reverse_feature_map, dataargs.rmRareF, true);
    }
  if (dataargs.rmRareL > 0)
    {
      data->removeRareLabels(label_map, reverse_label_map, dataargs.rmRareL, true);
    }
  
  if (dataargs.xnorm)
    {
      data->xstdnormal(mean, sdev, true, center, true);
    }
  if (dataargs.xunit)
    {
      data->xunitnormal();
    }
  data->xscale(dataargs.xscale);
  
  // done with modifying the data. Make it const.
  shared_ptr<const MCxyData> const_data = static_pointer_cast<const MCxyData>(data);
  data = nullptr;  // only use the const version from now on to make sure the data does not change. 
  
  MCsoln prev_soln;
  cout << "solfile" << solverargs.prev_soln_file << endl;
  if (solverargs.prev_soln_file.size() > 0)
    {
      ifstream is;
      is.open(solverargs.prev_soln_file);
      if (!is.good())
	{
	  throw runtime_error("Error opening previous solution file");
	}
      prev_soln.read(is);
    }

  MClearnFilter learner(const_data, prev_soln, solverargs.params);

  learner.learn();
  learner.saveSolution(outfile, outbinary?MCsoln::BINARY:MCsoln::TEXT);
}
