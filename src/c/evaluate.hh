#ifndef EVALUATE_HH
#define EVALUATE_HH
/** \file
 * inline template (not needed for library linking - libmcfilter includes DenseM & SparseM support)
 */

#include "filter.h"
#include "evaluate.h"
#include "predict.hh"   // oh, actually need the inline defn of predict for library
#include "utils.h"
#include "EigenIO.h"


template <typename EigenType> inline
void predict_chunk(predvec& predictions, VectorXsz& no_active,
		   doublevec& filter_time, doublevec& predict_time,
		   doublevec& total_time,
		   const EigenType& x, 
		   const DenseColM* wmat, const DenseColM* lmat,
		   const DenseColM* umat,		   
		   const DenseColMf& ovaW_chunk, 
		   const size_t start_class, predtype thresh, int k,
		   bool allproj, bool verbose)
{
  time_t start, stop;
  int pred_k = k>10?k:10;
  size_t n = x.rows();
  size_t noClasses = ovaW_chunk.cols();
  int nproj = wmat?wmat->cols():1;
  size_t nact;
 
  // if calculatin predictions for all projections, start from the first one. 
  // Else go directly to the last one 
  int np = allproj?nproj:1;
  assert(predictions.size() == np);
  assert(no_active.size() == np);
  assert(filter_time.size() == np);
  assert(predict_time.size() == np);
  assert(total_time.size() == np);

  assert(!umat||(umat->rows()==noClasses));
  assert(!lmat||(lmat->rows()==noClasses));

  if (!wmat)
    {
      // no filter, so everything is active
      no_active[0] += n*noClasses;
    }
      
  ActiveDataSet* active=NULL;

  // The predicions, times and number of active bits are stored in reverse order 
  // This way the results with all the filters in in the first position, result 
  // with only "sp" filters is in positions npredsets-1. 

  double ftime = 0;
  DenseM projections;

  for (int proj_no=0; proj_no <  nproj; proj_no++)
    {
      if(wmat)
	{
	  // init filter -- do not count this against the time 
	  if(verbose)
	    {
	      std::cout << "Initializing filter " << proj_no << std::endl;
	    }

	  VectorXd l = lmat->col(proj_no);
	  VectorXd u = umat->col(proj_no);
	  Filter f(l,u);	  

	  if(verbose)
	    {
	      std::cout << "Applying filter " << proj_no << std::endl;
	    }
	  time(&start);	  
	  VectorXd proj = x*wmat->col(proj_no);
	  nact = update_active(&active, f, proj);
	  time(&stop);
	  ftime += difftime(stop,start);

	  // I get the number of active classes here because 
	  // I want to know how many classes were filtered by each projection
	  // even if allproj is false. 
	  // this shoudl be turned off when running in production mode since
	  // counting in bitvectors is expensive (linear) 
	  
	  nact += n; // we had to do one dot product per example for the filter. 
	  no_active[nproj - proj_no - 1] += nact;
	}
      
      if (allproj || proj_no == nproj-1)
	{
	  filter_time[nproj - proj_no - 1] += ftime; 
	  
	  if(verbose)
	    {
	      std::cout << "Predicting..." << std::endl;
	    }
	  
	  time(&start);
	  predict(predictions[nproj - proj_no - 1], x, ovaW_chunk, active, nact, verbose, thresh, pred_k, start_class); 
	  time(&stop);
	  predict_time[nproj - proj_no - 1] += difftime(stop,start);
	  total_time[nproj - proj_no -1] += predict_time[nproj - proj_no - 1] + filter_time[nproj - proj_no - 1]; 
	}
    }
  if (active)
    {
      free_ActiveDataSet(active);
    }
}    


template <typename EigenType> inline
void evaluate_projection(const std::vector<EigenType*>& x, 
			 const std::vector<SparseMb*>& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 const std::vector<std::string>& setnames,
			 bool allproj, bool verbose, ostream& out /*=cout*/)
{
  
  assert(x.size() > 0);

  int nproj = wmat?wmat->cols():1;  
  int nrSets = x.size();

  std::vector<predvec*> predictions(nrSets);

  std::vector<doublevec*> total_time(nrSets);
  std::vector<doublevec*> filter_time(nrSets);
  std::vector<doublevec*> predict_time(nrSets);
  std::vector<VectorXsz*> no_active(nrSets);

  // initialization for each dataset
  for (int set=0;set < nrSets; set++)
    {  
      size_t n = x[set]->rows();

      total_time[set] = new doublevec(allproj?nproj:1);
      filter_time[set] = new doublevec(allproj?nproj:1);
      predict_time[set] = new doublevec (allproj?nproj:1);      
      no_active[set] = new VectorXsz(nproj);
      
      predictions[set] = new predvec(allproj?nproj:1);
      for (int proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  predictions[set]->at(nproj-proj_no) = new PredictionSet(n);	 
	}      
      std::fill(total_time[set]->begin(),total_time[set]->end(), 0.0);
      std::fill(filter_time[set]->begin(),filter_time[set]->end(), 0.0);
      std::fill(predict_time[set]->begin(),predict_time[set]->end(), 0.0);
      no_active[set]->setZero();      
    }
  
  // make preditions for each dataset
  for (int set=0;set < nrSets; set++)
    {  
      predict_chunk(*(predictions[set]), *(no_active[set]), *(filter_time[set]), *(predict_time[set]), *(total_time[set]), *(x[set]), wmat, lmat, umat, ovaW, 0, thresh, k, allproj, verbose);

    }
     

  //evaluate performances and write them to a file for each dataset
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }
  

  std::vector<double> MicroF1(allproj?nproj:1), MacroF1(allproj?nproj:1);
  std::vector<double> MacroF1_2(allproj?nproj:1);
  std::vector<double> MicroPrecision(allproj?nproj:1), MacroPrecision(allproj?nproj:1);
  std::vector<double> MicroRecall(allproj?nproj:1), MacroRecall(allproj?nproj:1);
  std::vector<double> Top1(allproj?nproj:1), Top5(allproj?nproj:1), Top10(allproj?nproj:1);
  std::vector<double> Prec1(allproj?nproj:1), Prec5(allproj?nproj:1), Prec10(allproj?nproj:1);

  for (int set=0;set < nrSets; set++)
    {
      for (int proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
	{      
	  predictions[set]->at(nproj-proj_no)->ThreshMetrics(MicroF1[nproj-proj_no], MacroF1[nproj-proj_no], MacroF1_2[nproj-proj_no], MicroPrecision[nproj-proj_no], MacroPrecision[nproj-proj_no], MicroRecall[nproj-proj_no], MacroRecall[nproj-proj_no], *(y[set]), thresh, k);
	  predictions[set]->at(nproj-proj_no)->TopMetrics(Prec1[nproj-proj_no], Top1[nproj-proj_no], Prec5[nproj-proj_no], Top5[nproj-proj_no], Prec10[nproj-proj_no], Top10[nproj-proj_no], *(y[set]));
	}      
      size_t n = x[set]->rows();
      size_t noClasses = y[set]->cols();
      size_t total_preds = n*noClasses;
      output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, 
		   MicroRecall, MacroRecall, 
		   Top1, Top5, Top10, Prec1, Prec5, Prec10, 
		   *(no_active[set]),  total_preds, *(total_time[set]), *(filter_time[set]), *(predict_time[set]), projname + " " + setnames[set] + "_", out);
    }
    
  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }

  // cleanup for each dataset
  for (int set=0;set < nrSets; set++)
    {  
      for (int proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  delete predictions[set]->at(nproj-proj_no);
	}
      delete predictions[set];      
      delete total_time[set];
      delete filter_time[set];
      delete predict_time[set];
      delete no_active[set];      
    }
}

template <typename EigenType> inline
void evaluate_projection_chunks(const std::vector<EigenType*>& x, 
				const std::vector<SparseMb*>& y, 
				const string& ova_file, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				const std::vector<std::string>& setnames,
				bool allproj, bool verbose, ostream& out /*=cout*/)
{
  assert(chunks > 0);
  assert(x.size() > 0);
  int nproj = wmat?wmat->cols():1;  
  int nrSets = x.size();

  std::vector<predvec*> predictions(nrSets);

  std::vector<doublevec*> total_time(nrSets);
  std::vector<doublevec*> filter_time(nrSets);
  std::vector<doublevec*> predict_time(nrSets);
  std::vector<VectorXsz*> no_active(nrSets);

  std::vector<double> MicroF1(allproj?nproj:1), MacroF1(allproj?nproj:1);
  std::vector<double> MacroF1_2(allproj?nproj:1);
  std::vector<double> MicroPrecision(allproj?nproj:1), MacroPrecision(allproj?nproj:1);
  std::vector<double> MicroRecall(allproj?nproj:1), MacroRecall(allproj?nproj:1);
  std::vector<double> Top1(allproj?nproj:1), Top5(allproj?nproj:1), Top10(allproj?nproj:1);
  std::vector<double> Prec1(allproj?nproj:1), Prec5(allproj?nproj:1), Prec10(allproj?nproj:1);

  // initialization for each dataset
  for (int set=0;set < nrSets; set++)
    {  
      size_t n = x[set]->rows();

      total_time[set] = new doublevec(allproj?nproj:1);
      filter_time[set] = new doublevec(allproj?nproj:1);
      predict_time[set] = new doublevec (allproj?nproj:1);      
      no_active[set] = new VectorXsz(nproj);
      
      predictions[set] = new predvec(allproj?nproj:1);
      for (int proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  predictions[set]->at(nproj-proj_no) = new PredictionSet(n);	 
	}      
      std::fill(total_time[set]->begin(),total_time[set]->end(), 0.0);
      std::fill(filter_time[set]->begin(),filter_time[set]->end(), 0.0);
      std::fill(predict_time[set]->begin(),predict_time[set]->end(), 0.0);
      no_active[set]->setZero();      
    }

  
  size_t dim = x[0]->cols();
  size_t noClasses = y[0]->cols();
  size_t start_class = 0;  
  { // have an internal block so that ovaW goes out of scope at the end of it 
    // and memory is released
    DenseColMf ovaW;
    DenseColM lmat_chunk;
    DenseColM umat_chunk;
    
    for (int chunk = 0; chunk < chunks; chunk++)
      {
	size_t chunk_size = noClasses/chunks + (chunk < (noClasses % chunks));
	if (verbose)
	  {
	    cout << "Load chunk ... " << endl;
	  }
	read_binary(ova_file.c_str(), ovaW, dim, chunk_size, start_class);
	if (verbose)
	  {
	    cout << "Done load chunk. " << endl;
	  }
	
	
	if (wmat)
	  {
	    // this copies data
	    lmat_chunk = lmat->block(start_class,0,chunk_size,lmat->cols());
	    umat_chunk = umat->block(start_class,0,chunk_size,umat->cols());
	  }		

	// make preditions for each dataset	
	for (int set=0; set < nrSets; set++)
	  {  	    
	    predict_chunk(*(predictions[set]), *(no_active[set]), *(filter_time[set]), *(predict_time[set]), *(total_time[set]), *(x[set]), wmat, &lmat_chunk, &umat_chunk, ovaW, start_class, thresh, k, allproj, verbose);	
	  }
	start_class = start_class+chunk_size;           
      }
  } // we don't need ovaW any more
  assert(start_class == noClasses);
  

  //evaluate performances and write them to a file for each dataset
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }
  
  for (int set=0;set < nrSets; set++)
    {
      for (int proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
	{      
	  predictions[set]->at(nproj-proj_no)->ThreshMetrics(MicroF1[nproj-proj_no], MacroF1[nproj-proj_no], MacroF1_2[nproj-proj_no], MicroPrecision[nproj-proj_no], MacroPrecision[nproj-proj_no], MicroRecall[nproj-proj_no], MacroRecall[nproj-proj_no], *(y[set]), thresh, k);
	  predictions[set]->at(nproj-proj_no)->TopMetrics(Prec1[nproj-proj_no], Top1[nproj-proj_no], Prec5[nproj-proj_no], Top5[nproj-proj_no], Prec10[nproj-proj_no], Top10[nproj-proj_no], *(y[set]));
	}      
      size_t n = x[set]->rows();
      assert(noClasses == y[set]->cols());
      size_t total_preds = n*noClasses;
      output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, 
		   MicroRecall, MacroRecall, 
		   Top1, Top5, Top10, Prec1, Prec5, Prec10, 
		   *(no_active[set]),  total_preds, *(total_time[set]), *(filter_time[set]), *(predict_time[set]), projname + " " + setnames[set] + "_", out);
    }
  
  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }
  // cleanup for each dataset
  for (int set=0;set < nrSets; set++)
    {  
      for (int proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  delete predictions[set]->at(nproj-proj_no);
	}
      delete predictions[set];      
      delete total_time[set];
      delete filter_time[set];
      delete predict_time[set];
      delete no_active[set];      
    }
  
}




template <typename EigenType> inline
void get_projection_measures(const EigenType& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue)
{
  size_t n = x.rows();
  size_t noClasses = y.cols();
  int nproj = wmat.cols();

  ActiveDataSet* active=NULL;

  VectorXsz no_active(nproj); // don't need this

  nrTrueActive.setZero(noClasses);
  nrActive.setZero(noClasses);
  nrTrue.setZero(noClasses);

  active = getactive (no_active, x, wmat, lmat, umat, verbose);

  for (size_t i = 0; i < n; i++)
    {
      for (SparseMb::InnerIterator it(y,i); it; ++it)
	{
	  if((*(active->at(i)))[it.col()])
	    {
	      nrTrueActive[it.col()]++;
	    }
	  nrTrue[it.col()]++;
	}
      for (size_t j = 0; j<noClasses; j++)
	{
	  nrActive[j] += (*(active->at(i)))[j];
	}
    }
}

#endif // EVALUATE_HH
