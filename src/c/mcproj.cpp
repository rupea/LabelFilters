/** \file
 * For every example in a test set, use <B>M</B>ulti-<B>C</B>lass <B>proj</B>ections
 * to determine {possible class labels}.
 */

#include "parameter-args.h"
#include "mclinearClass.h"
#include "linearModel.h"
#include "mcxydata.h"
#include "mcfilter.h"
#include <boost/program_options.hpp>
#include <iostream>

using namespace std;
namespace po=boost::program_options;
using namespace po;

#define PRINT_PERF(PERF) do {out<<#PERF << ": ";  for (int i = 0; i < perfs.size(); i++) {out << perfs[i].PERF << " ";} out<<endl;} while(0)
void outputPerfs(ostream& out, string const& modelfile, string const& datafile, string const& filterfile, vector<int> const& nProj, vector<PerfStruct> const& perfs)
{
  out << "Model: " << modelfile << endl;
  out << "Data: " << datafile << endl;
  out << "Filter: "<< filterfile << endl;

  assert (nProj.size() == perfs.size());

  out << "nProj: ";
  for (int i = 0; i < nProj.size(); i++)
    {
      out << nProj[i] << " ";
    }
  out << endl;

  PRINT_PERF(prcFeasible);
  PRINT_PERF(Prec1);
  PRINT_PERF(Prec5);
  PRINT_PERF(Prec10);
  PRINT_PERF(Top1);
  PRINT_PERF(Top5);
  PRINT_PERF(Top10);
  PRINT_PERF(MacroPrecision);
  PRINT_PERF(MacroRecall);
  PRINT_PERF(MacroF1);
  PRINT_PERF(MacroF1_2);
  PRINT_PERF(MicroPrecision);
  PRINT_PERF(MicroRecall);
  PRINT_PERF(MicroF1);
}
#undef PRINT_PERF


int main(int argc, char**argv){
#ifndef NDEBUG
  int const verb = +1;        // verbosity modifier
#else
  int const verb = 0;         // verbosity modifier
#endif
  
  po::variables_map vm;

  vector<PerfStruct> perfs;
  PerfStruct nofilter_perfs;
  bool nofilter_ok = false;
  int verbose = 0;
  
  try {
    po::options_description desc;
    po::options_description general("Genral Options");      
    
    general.add_options()      
      ("verbose,v",value<int>()->default_value(1),"Verbosity level")
#ifdef _OPENMP
      ("nthreads", value<int>()->default_value(0),"Number of threads to use. 0 means all available")
#endif
      ("help,h","This help")
      ;
    
    desc.add(general).add(MCxyDataArgs::getDesc()).add(MCprojectorArgs::getDesc()).add(MCclassifierArgs::getDesc());
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

#ifdef _OPENMP
  if (vm.count("nthreads"))
    {
      int nthreads = vm["nthreads"].as<int>();      
      if (nthreads > 0)
	{
	  omp_set_num_threads(nthreads);
	}
    }
#endif
  
  MCxyDataArgs dataargs(vm);
  MCprojectorArgs filterargs(vm);
  MCclassifierArgs modelargs(vm);

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
  
      
  for (vector<string>::iterator mit = modelargs.modelFiles.begin(); mit != modelargs.modelFiles.end(); ++mit)
    {
      // prepare the classifier 
      
      shared_ptr<linearModel> model = make_shared<linearModel>();
      model->read(*mit);
      // done with changing the model. Make it const
      shared_ptr<const linearModel> const_model = static_pointer_cast<const linearModel>(model);
      model = nullptr;  // only use the const version from now on to make sure the model does not change. 

      for (int dta = 0; dta < dataargs.xFiles.size(); ++dta)
	{
	  // prepare the data
	  string xfile = dataargs.xFiles[dta];
	  string yfile;
	  if (dataargs.yFiles.size() == 0)
	    {
	      // no yfile
	      yfile = "";
	    }
	  else if (dataargs.yFiles.size() == 1)
	    {
	      // only one label file for all datasets
	      yfile = dataargs.yFiles[0];
	    }
	  else if (dataargs.yFiles.size() == dataargs.yFiles.size())
	    {
	      yfile = dataargs.yFiles[dta];
	    }
	  else
	    {
	      throw runtime_error("xFiles and yFiles do not match");
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
	  
	  if (filterargs.lfFiles.size() == 0)
	    {
	      filterargs.lfFiles.push_back("");
	    }
	  for (int l=0; l<filterargs.lfFiles.size(); l++)
	    {
	      // prepare filter.

	      shared_ptr<MCfilter> lf = make_shared<MCfilter>();
	      if (filterargs.lfFiles[l].size())
		{
		  lf->read(filterargs.lfFiles[l]);
		}
	      
	      //done with modifying the label filter. Make it const
	      shared_ptr<const MCfilter> const_lf = static_pointer_cast<const MCfilter>(lf);
	      
	      // we have the model, data and filter. We can predict/evaluate models using a linear classifier		
	      MClinearClassifier classifier(const_data, const_model, const_lf, verbose);
	      // set keep_top, keep_thresh
	      classifier.setPruneParams(modelargs.keep_thresh, modelargs.keep_top);

	      // replace -2 and -1 nProj arguments with the correct values.
	      if (std::find(filterargs.nProj.begin(), filterargs.nProj.end(), -2) != filterargs.nProj.end())
		{
		  filterargs.nProj.clear();
		  for (int i = 0; i < const_lf->nFilters(); i++)
		    {
		      filterargs.nProj.push_back(i);
		    }
		}
	      else
		{
		  vector<int>::iterator it = std::find(filterargs.nProj.begin(), filterargs.nProj.end(), -1);		   
		  if ( it != filterargs.nProj.end())
		    {
		      *it = const_lf->nFilters();
		    }
		}
		  
	      // set the nubmer of filters to apply
	      for (vector<int>::iterator npit=filterargs.nProj.begin(); npit != filterargs.nProj.end(); ++npit)
		{
		  int np = *npit;

		  if (np == 0 && nofilter_ok)
		    {
		      //the results for 0 projections or no filters are the same regardelss of the label filter used.
		      // So used saved results if available and save them if not
		      perfs.push_back(nofilter_perfs);
		    }
		  else
		    {
		  
		      if (np > const_lf->nFilters())
			{
			  throw runtime_error(to_string(np) + " filters requested, but only " + to_string(const_lf->nFilters()) + " exist");
			}
		      
		      classifier.nProj(np);
		      classifier.predict();
		      perfs.push_back(classifier.evaluate(modelargs.threshold, modelargs.min_labels));
		      if (np == 0)
			{
			  //the results for 0 projections or no filters are the same regardelss of the label filter used.
			  // And they are expensive co calculate. So save reslults for late use
			  nofilter_perfs = perfs.back();
			  nofilter_ok = true;
			}
		    }		      
		}
	      
	      // Done. Output the performances 
	      outputPerfs(cout, *mit, dataargs.xFiles[dta], filterargs.lfFiles[l], filterargs.nProj, perfs);
	      
	    }
	  nofilter_ok = false; // data or model has changed, so result with no filter are not valid any more
	}
    }
}	    
