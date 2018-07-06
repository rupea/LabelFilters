
#include "parameter-args.h"
#include "parameter.h"
#include <iostream>
#include <cstdint>
#include <sstream>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>    // split_unix

using namespace std;
namespace po = boost::program_options;

using namespace po;


MCsolveArgs::MCsolveArgs():
  params(),
  prev_soln_file("")
{}

MCsolveArgs::MCsolveArgs(po::variables_map const& vm):
  MCsolveArgs()
{
  this->extract(vm);
}
  

po::options_description MCsolveArgs::getDesc()
{
  po::options_description desc("Solver Options");
  po::options_description main("Main Solver Options");
  main.add_options()
    ("C1", value<double>()->required(), "penalty for label outside correct (l,u)")
    ("C2", value<double>()->required(), "penalty for label inside incorrect (l,u)")
    ("nfilters", value<uint32_t>(), "# of label filters [5]")
    ("maxiter", value<uint32_t>(), "max iterations per projection. [1e8/batch_size]")
    ("eta0", value<double>(), "initial learning rate [0.1]")
    ("seed", value<uint32_t>(), "random nmber seed. 0 - initizize rn using current time, 1 - do not initialize rng [0]")
    ("prevsoln", value<string>(), "file with solution to be used by resume, reoptlu and/or initWtype=PREV")
    ("resume", value<bool>()->implicit_value(true), "add more filters to a previous solution")
    ("reoptlu", value<bool>()->implicit_value(true), "reoptimize the intevals [l,u] for previously learned filter directions.")
    ("sample", value<uint32_t>()
     , "how many negative classes to use for gradient estimate.\nTry 100 or 1000 to speed up. 0 ~ all [0]")
    ("threads", value<uint32_t>(), "# of threads, 0 ~ use OMP_NUM_THREADS")
    ;
  po::options_description dev("Advanced Solver Options");
  dev.add_options()
    ("update,u", value<std::string>(), "PROJECTED | MINIBATCH : gradient update type\n   PROJECTED requires batchsize of 1")
    ("batchsize,b", value<uint32_t>(), "batch size. 0~full gradient, [1 if PROJECTED,  1000 if MINIBATCH]")
    ("avgrad", value<bool>()->implicit_value(true), "Use averaged gradient [true]")
    ("avgradstart", value<uint32_t>(), "averaging gradient start iteration [max(nExamples,dim)]")
    ("etatype", value<std::string>()
     , "CONST | SQRT | LIN | 3_4 : learning rate schedule [3_4 if using averaged gradiend, LIN if not]")
    ("etamin", value<double>(), "minimum learning rate limit [0.0]")
    ("toptlu", value<uint32_t>(), "expensive exact {l,u} optmization period. 0 for no optimization. [once, at maxiter]")
    ("treorder", value<uint32_t>(), "reorder iteration period [1000]")
    ("reorderby", value<std::string>()
     , "Permutation re-ordering: PROJ ~ mean of projected instances | MID ~ range midpoints.")
    ("treport", value<uint32_t>(), "report the objective value every treport iteratoins. 0 for no reporting. [maxiter/10]")
    ("remove_constraints", value<bool>()->implicit_value(true)
     , "after each projection, remove constraints involving incorrect labels already eliminated for the example [true]")
    ("remove_class_constraints", value<bool>()->implicit_value(true)
     , "after each projection, remove constraints involving correct labels already eliminated for the example [false]")
    ("adjustC", value<bool>()->implicit_value(true)
     , "adjust C1 and C2 to acount for the removed constraints [true]")
    ("initWtype", value<string>(), "How to initialize filter direction w [DIFF]. Options are:\n  DIFF ~ difference between centers of two random classes\n  ZERO ~ zero vector\n  RAND ~ random\n  PREV ~ from a given vector (given as a prevoius solution). Incompatible with resume or reoptlu.")
    ("initWnorm", value<double>(), "Norm of the initial filter direction. Used only for initWtype=DIFF|RAND. [10]")
    ("init_orthogonal", value<bool>()->implicit_value(true), "UNTESTED - Initialize new filter direction orthogonal on previous filter directions. Used only for initWtype = RAND. [false]")
    ("wt_by_nclasses", value<bool>()->implicit_value(true), "UNTESTED - weight example by the inverse of the number of labels it has when considering outside incorrect interval constraints (C2->C2/#labels) [false]")
    ("wt_class_by_nclasses", value<bool>()->implicit_value(true), "UNTESTED - weight example by the inverse of the number of labels it has when considering inside correct interval constraints (C1->C1/#labels) [false]")
    ;
  // Add generic, main and development option groups	  
  desc.add(main).add(dev);
  
#if GRADIENT_TEST /*|| others?*/
  po::options_description compiled("Compile-time options (enabled by compile flags)");
#if GRADIENT_TEST
  compiled_add_options()
    ("tgradtest", value<uint32_t>(), "iter period for finite difference gradient test[1]")
    ("ngradtest", value<uint32_t>(), "directions per gradient test [1000]")
    ("deltagradtest", value<double>(), "step size per gradient test [1e-4]")
    ;
#endif
  desc.add(compiled);
#endif
  return desc;
}

  
void MCsolveArgs::extract(po::variables_map const& vm)
{
  if (vm.count("C1"))
    {
      params.C1(vm["C1"].as<double>());
    }
  if (vm.count("C2"))
    {
      params.C2(vm["C2"].as<double>());
    }
  if (vm.count("nfilters"))
    {
      params.nfilters(vm["nfilters"].as<uint32_t>());
    }
  if (vm.count("maxiter"))
    {
      params.max_iter(vm["maxiter"].as<uint32_t>());
    }
  if (vm.count("eta0"))
    {
      params.eta(vm["eta0"].as<double>());
    }
  if (vm.count("seed"))
    {
      params.seed(vm["seed"].as<uint32_t>());
    }
  if (vm.count("prevsoln"))
    {
      prev_soln_file = vm["prevsoln"].as<string>();
    }  
  if (vm.count("resume"))
    {
      params.resume(vm["resume"].as<bool>());
    }
  if (vm.count("reoptlu"))
    {
      params.reoptimize_LU(vm["reoptlu"].as<bool>());
    }
  if (vm.count("sample"))
    {
      params.class_samples(vm["sample"].as<uint32_t>());
    }
  // if (vm.count("threads"))
  //   {
  //     params.num_threads(vm["threads"].as<uint32_t>());
  //   }
  //update before batchsize
  if (vm.count("update"))
    {
      Update_Type ut;
      fromstring(vm["update"].as<string>(), ut);
      params.update_type(ut);
    }
  if (vm.count("batchsize"))
    {
      params.batch_size(vm["batchsize"].as<uint32_t>());
    }  
  if (vm.count("avgrad"))
    {
      params.averaged_gradient(vm["avgrad"].as<bool>());
    }
  if (vm.count("avgradstart"))
    {
      params.avg_epoch(vm["avgradstart"].as<uint32_t>());
    }
  if (vm.count("etatype"))
    {
      Eta_Type et;
      fromstring(vm["etatype"].as<string>(), et);
      params.eta_type(et);
    }
  if (vm.count("etamin"))
    {
      params.min_eta(vm["etamin"].as<double>());
    }
  if (vm.count("toptlu"))
    {
      params.optimizeLU_epoch(vm["toptlu"].as<uint32_t>());
    }
  if (vm.count("treorder"))
    {
      params.reorder_epoch(vm["treorder"].as<uint32_t>());
    }
  if (vm.count("reorderby"))
    {
      Reorder_Type rt;
      fromstring(vm["reorderby"].as<string>(), rt);
      params.reorder_type(rt);
    }
  if (vm.count("treport"))
    {
      params.report_epoch(vm["treport"].as<uint32_t>());
    }
  if (vm.count("remove_constraints"))
    {
      params.remove_constraints(vm["remove_constraints"].as<bool>());
    }
  if (vm.count("remove_class_constraints"))
    {
      params.remove_class_constraints(vm["remove_class_constraints"].as<bool>());
    }
  if (vm.count("adjustC"))
    {
      params.adjust_C(vm["adjustC"].as<bool>());
    }
  // initWtype before initWnorm and init_orthogonal
  if (vm.count("initWtype"))
    {
      Init_W_Type it;
      fromstring(vm["initWtype"].as<string>(), it);
      params.init_type(it);
    }
  if (vm.count("initWnorm"))
    {
      params.init_norm(vm["initWnorm"].as<double>());
    }
  if (vm.count("init_orthogonal"))
    {
      params.init_orthogonal(vm["init_orthogonal"].as<bool>());
    }  
  if (vm.count("wt_by_nclasses"))
    {
      params.ml_wt_by_nclasses(vm["wt_by_nclasses"].as<bool>());
    }
  if (vm.count("wt_class_by_nclasses"))
    {
      params.ml_wt_class_by_nclasses(vm["wt_class_by_nclasses"].as<bool>());
    }

#if GRADIENT_TEST  // Off, by default, at compile time
  if (vm.count("tgradtest"))
    {
      params.finite_diff_test_epoch(vm["tgradtest"].as<uint32_t>());
    }
  if (vm.count("ngradtest"))
    {
      params.no_finite_diff_tests(vm["ngradtest"].as<uint32_t>());
    }
  if (vm.count("deltagradtest"))
    {
      params.finite_diff_test_delta(vm["deltagradtest"].as<double>());
    }
#endif  
}



//
// ----------------------- MCprojArgs --------------------
//

po::options_description MCprojectorArgs::getDesc(){
  po::options_description projopt("Label Filter Options");
  projopt.add_options()
    ("lfFiles,f", value<std::vector<std::string>>()->multitoken()->zero_tokens()->composing(), "label filter files") 
    ("nProj", value<std::vector<int>>()->multitoken()->composing(), "number of filters to apply. 0 = no filters, -1 = all filters. -2 = {0,1,2,..., all filters}")
    ;
  return projopt;
}

MCprojectorArgs::MCprojectorArgs()
  : 
  lfFiles()	    
  , nProj({-1})
{}
  
MCprojectorArgs::MCprojectorArgs(po::variables_map const& vm)
  : MCprojectorArgs()
{
  this->extract(vm);
}
  
void MCprojectorArgs::extract(po::variables_map const& vm)
{
  if (vm.count("lfFiles"))
    {
      lfFiles = std::vector<std::string>(vm["lfFiles"].as<std::vector<std::string>>());
    }
    
  if (vm.count("nProj"))
    {
      nProj = std::vector<int>(vm["nProj"].as<std::vector<int>>());
    }
}
  
  
  
po::options_description MCxyDataArgs::getDesc(){
  po::options_description dataopt("Data Options");
  dataopt.add_options()
    ("xFiles,x", value<std::vector<std::string>>()->multitoken()->composing()->required(), "Data files. LibSVM, XML or binary dense/sparse formats. LibSVM and XML formats also contain the labels. Binary formats don't contain labels.") 
    ("yFiles,y", value<std::vector<std::string>>()->multitoken()->zero_tokens()->composing(), "Lables for the data. Must be one file used with all xFiles, or the same number and in same order as xFiles.")
    ("rmRareF", value<uint>()->default_value(0U), "Remove features with fewer than this many non-zero values.")
    ("rmRareL", value<uint>()->default_value(0U), "Remove labels that appear fewer than this many times.")
    ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "Col-normalize the data(mean=0, stdev=1)")
    ("center", value<bool>()->implicit_value(true), "Remove the mean when col-normalizing. Default true for dense data and false for sparse data.")
    ("normdata", value<std::string>()->default_value(""), "Data to compute statistics for col-normalization and/or for rare feature and rare labels remova. If empty, statistis are calculated on the first xFiles argument")
    ("xunit", value<bool>()->implicit_value(true)->default_value(false), "Row-normalize data to unit length")
    ("xscale", value<double>()->default_value(1.0), "scale each x example. rare feature removal, xnorm, xunit, xscal applied in order.")
    ;
  return dataopt;
}
  
MCxyDataArgs::MCxyDataArgs()
  : xFiles()
  , normData()
  , rmRareF(0U)
  , rmRareL(0U)
  , xnorm(false)
  , center(-1)
  , xunit(false)
  , xscale(1.0)
{}
  
MCxyDataArgs::MCxyDataArgs(po::variables_map const& vm)
  : MCxyDataArgs()
{
  this->extract(vm);
}
  
void MCxyDataArgs::extract(po::variables_map const& vm)
{ 
  if (vm.count("xFiles"))
    {
      xFiles = std::vector<std::string>(vm["xFiles"].as<std::vector<std::string>>());
    }
  if (vm.count("yFiles"))
    {
      yFiles = std::vector<std::string>(vm["xFiles"].as<std::vector<std::string>>());
    }
    
  if (yFiles.size()!=0 && yFiles.size()!=1 && yFiles.size() != xFiles.size())
    {
      throw std::runtime_error("Different numbers of xFiles and yFiles");
    }

  if (vm.count("rmRareF"))
    {
      rmRareF = vm["rmRareF"].as<uint>();
    }

  if (vm.count("rmRareL"))
    {
      rmRareL = vm["rmRareL"].as<uint>();
    }

  if (vm.count("xnorm"))
    {
      xnorm = vm["xnorm"].as<bool>();
    }

  if (vm.count("center"))
    {
      center = vm["center"].as<bool>()?1:0;
    }

  if (vm.count("normdata"))
    {
      normData = vm["normdata"].as<std::string>();
    }    
  if (vm.count("xunit"))
    {
      xunit = vm["xunit"].as<bool>();
    }
  if (vm.count("xscale"))
    {
      xunit = vm["xscale"].as<double>();
    }
}

po::options_description MCclassifierArgs::getDesc(){
  po::options_description classopt("Classifier Options");
  classopt.add_options()
    ("modelFiles", value<std::vector<std::string>>()->multitoken()->composing()->required(), "Files with saved models. Binary dense/sparse format or text sparse format")       
    ("keep_top", value<uint32_t>()->default_value(10), "Keep at least keep_top predictions for each instance. Others are considered -inf. (i.e. can not calculate Prec@20 if keep_top < 20")
    ("keep_thresh", value<double>()->default_value(0.0), "Keep all predictions larger than keep_thresh. Others are considered -inf")
    ("threshold", value<double>()->default_value(0.0), "Threshold to use when assigning labels to an instance. Used when calculating precision/recall")
    ("min_labels", value<uint32_t>()->default_value(1), "Each example will have at least min_labels assigned (even if they are below the threshold). Used when calculating precision/recall") 
    ;
  return classopt;
}

MCclassifierArgs::MCclassifierArgs()
  : modelFiles()
  , keep_thresh(0.0)
  , keep_top(10)
  , threshold(0.0)
  , min_labels(1)
{}

MCclassifierArgs::MCclassifierArgs(po::variables_map const& vm)
  : MCclassifierArgs()
{
  this->extract(vm);
}

void MCclassifierArgs::extract(po::variables_map const& vm)
{ 
  if (vm.count("modelFiles"))
    {
      modelFiles = std::vector<std::string>(vm["modelFiles"].as<std::vector<std::string>>());
    }
    
  if (vm.count("keep_top"))
    {
      keep_top = vm["keep_top"].as<uint32_t>();
    }
  if (vm.count("min_labels"))
    {
      min_labels = vm["min_labels"].as<uint32_t>();
    }
    
  if (min_labels > keep_top)
    {
      throw std::runtime_error("Asked to predict at least " + std::to_string(min_labels) + " per example, but keep_top is set to " + std::to_string(keep_top));
    }
    
  if (vm.count("keep_thresh"))
    {
      keep_thresh = vm["keep_thresh"].as<double>();
    }
  if (vm.count("threshold"))
    {
      threshold = vm["threshold"].as<double>();
    }
  if (threshold < keep_thresh)
    {
      throw std::runtime_error("Asked to use threshold " + std::to_string(threshold) + " but keep_thresh is set to " + std::to_string(keep_thresh));
    }    
}

