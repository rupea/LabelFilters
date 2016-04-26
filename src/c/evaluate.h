#ifndef _EVALUATE_H
#define _EVALUATE_H

#include "predict.h"
#include "utils.h"
#include "EigenIO.h"

//using namespace std;

typedef std::vector<PredictionSet*> predvec;
typedef std::vector<double> doublevec;

void output_perfs(const std::vector<double>& MicroF1, const std::vector<double>&  MacroF1,
		  const std::vector<double>& MacroF1_2, 
		  const std::vector<double>& MicroPrecision,
		  const std::vector<double>& MacroPrecision, 
		  const std::vector<double>& MicroRecall, 
		  const std::vector<double>& MacroRecall,
		  const std::vector<double>& Top1, const std::vector<double>& Top5,
		  const std::vector<double>& Top10, const std::vector<double>& Prec1, 
		  const std::vector<double>& Prec5, const std::vector<double>& Prec10,
		  const VectorXsz& nact, size_t total_preds, 
		  const std::vector<double>& total_time, 
		  const std::vector<double>& filter_time, 
		  const std::vector<double>& predict_time, 
		  string str="", ostream& out = cout)
{
  int proj;
  if (MicroF1.size() > 1)
    {
      out << str << "MicroF1_per_proj  ";
      for (proj = MicroF1.size() - 1; proj >=0 ; proj--)
	{
	  out << MicroF1[proj] << "  ";
	}
      out << endl;
    }
  if (MacroF1.size() > 1)
    {
      out << str << "MacroF1_per_proj  ";
      for (proj = MacroF1.size()-1;proj >=0 ; proj--)
	{
	  out << MacroF1[proj] << "  ";
	}
      out << endl;
    }
  if (MacroF1_2.size() > 1)
    {
      out << str << "MacroF1_2_per_proj  ";
      for (proj = MacroF1_2.size()-1;proj >=0 ; proj--)
	{
	  out << MacroF1_2[proj] << "  ";
	}
      out << endl;
    }
  if (MicroPrecision.size() > 1)
    {
      out << str << "MicroPrecision_per_proj  ";
      for (proj = MicroPrecision.size()-1; proj >=0; proj--)
	{
	  out << MicroPrecision[proj] << "  ";
	}
      out << endl;
    }
  if (MacroPrecision.size() > 1)
    {
      out << str << "MacroPrecision_per_proj  ";
      for (proj = MacroPrecision.size()-1;proj >=0 ; proj--)
	{
	  out << MacroPrecision[proj] << "  ";
	}
      out << endl;
    }
  if (MicroRecall.size() > 1)
    {
      out << str << "MicroRecall_per_proj  ";
      for (proj = MicroRecall.size()-1;proj >=0 ; proj--)
	{
	  out << MicroRecall[proj] << "  ";
	}
      out << endl;
    }
  if (MacroRecall.size() > 1)
    {
      out << str << "MacroRecall_per_proj  ";
      for (proj = MacroRecall.size()-1;proj >=0 ; proj--)
	{
	  out << MacroRecall[proj] << "  ";
	}
      out << endl;
    }
  if (Top1.size() > 1)
    {
      out << str << "Top1_per_proj  ";
      for (proj = Top1.size()-1;proj >=0 ; proj--)
	{
	  out << Top1[proj] << "  ";
	}
      out << endl;
    }
  if (Prec1.size() > 1)
    {
      out << str << "Prec1_per_proj  ";
      for (proj = Prec1.size()-1;proj >=0 ; proj--)
	{
	  out << Prec1[proj] << "  ";
	}
      out << endl;
    }
  if (Top5.size() > 1)
    {
      out << str << "Top5_per_proj  ";
      for (proj = Top5.size()-1;proj >=0 ; proj--)
	{
	  out << Top5[proj] << "  ";
	}
      out << endl;
    }
  if (Prec5.size() > 1)
    {
      out << str << "Prec5_per_proj  ";
      for (proj = Prec5.size()-1;proj >=0 ; proj--)
	{
	  out << Prec5[proj] << "  ";
	}
      out << endl;
    }
  if (Top10.size() > 1)
    {
      out << str << "Top10_per_proj  ";
      for (proj = Top10.size()-1;proj >=0 ; proj--)
	{
	  out << Top10[proj] << "  ";
	}
      out << endl;
    }
  if (Prec10.size() > 1)
    {
      out << str << "Prec10_per_proj  ";
      for (proj = Prec10.size()-1;proj >=0 ; proj--)
	{
	  out << Prec10[proj] << "  ";
	}
      out << endl;
    }

  // the number of active classes is always calculated
  out << str << "active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << nact[proj] << "  ";
    }
  out << endl;
  out << str << "prc_active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << nact[proj]*1.0/total_preds << "  ";
    }
  out << endl;
  out << str << "speedup_active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << total_preds*1.0/nact[proj] << "  ";
    }
  out << endl;

  out << str << "MicroF1  " << MicroF1[0] << endl;
  out << str << "MacroF1  " << MacroF1[0] << endl;
  out << str << "MacroF1_2  " << MacroF1_2[0] << endl;
  out << str << "MicroPrecision  " << MicroPrecision[0] << endl;
  out << str << "MacroPrecision  " << MacroPrecision[0] << endl;
  out << str << "MicroRecall  " << MicroRecall[0] << endl;
  out << str << "MacroRecall  " << MacroRecall[0] << endl;
  out << str << "Top1  " << Top1[0] << endl;
  out << str << "Prec1  " << Prec1[0] << endl;
  out << str << "Top5  " << Top5[0] << endl;
  out << str << "Prec5  " << Prec5[0] << endl;
  out << str << "Top10  " << Top10[0] << endl;
  out << str << "Prec10  " << Prec10[0] << endl;
  out << str << "nactive  " << nact(0) << endl;
  out << str << "prc_active  " << nact(0)*1.0/total_preds << endl;
  out << str << "speedup  " << total_preds*1.0/nact(0) << endl;
  out << str << "total  " << total_preds << endl;
  out << str << "total_time  " << total_time[0] << endl;
  out << str << "filter_time  " << filter_time[0] << endl;
  out << str << "predict_time  " << predict_time[0] << endl;
}

template <typename EigenType>
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
  time_t start_filter, start_predict, stop;
  int pred_k = k>10?k:10;
  size_t n = x.rows();
  size_t noClasses = ovaW_chunk.cols();
  int nproj = wmat?wmat->cols():1;
  size_t nact;
 
  // if calculatin predictions for all projections, start from the first one. 
  // Else go directly to the last one 
  int sp = allproj?1:nproj; 
  int np = nproj-sp+1; 
  assert(predictions.size() == np);
  assert(no_active.size() = np);
  assert(filter_time.size() = np);
  assert(predict_time.size() = np);
  assert(total_time.size() = np);

  assert(!umat||(umat.rows()==noClasses));
  assert(!lmat||(lmat.rows()==noClasses));

  if (!wmat)
    {
      // no filter, so everything is active
      no_active[0] += n*noClasses;
    }
      

  VectorXsz no_active_chunk(np);
  ActiveDataSet* active=NULL;

  // The predicions, times and number of active bits are stored in reverse order 
  // This way the results with all the filters in in the first position, result 
  // with only "sp" filters is in positions npredsets-1. 

  // this does a lot of extra work by refiltersin with fewer projections, but I won't 
  // worry about it now. Usually this functions will be called with allproj=false. 
  for (int proj_no=sp; proj_no <= nproj; proj_no++)
    {
      DenseColM wmat_truncated;
      if (wmat)
	{
	  wmat_truncated = wmat->leftCols(proj_no);
	}
      if (verbose)
	{
	  cout << "Filter chunk ... " << endl;
	}
      time(&start_filter);
      if (wmat)
	{
	  // this initializes the filter every time. A production implementation
	  // would only initialize the filter once, then pass all the data through it. 
	  active = getactive (no_active_chunk, x, wmat_truncated, *lmat, *umat, verbose);
	  if (proj_no == nproj)
	    {
	      // I get the number of active classes from getactive because 
	      // I want to know how many classes were filtered by each projection
	      // even if allproj is false. 
	      // this shoudl be turned off when running in production mode since
	      // counting in bitvectors is expensive (linear) 
	      no_active += no_active_chunk.reverse();
	    }
	}
      time(&start_predict);
      filter_time[nproj-proj_no] += difftime(start_predict,start_filter);	   
      if (verbose)
	{
	  cout << "Done filter chunk." << endl;
	  cout << "Predict chunk ... " << endl;
	}
      predict(predictions[nproj-proj_no], x, ovaW_chunk, active, nact, verbose, thresh, pred_k, start_class); 
      
      if (verbose)
	{
	  cout << "Done predict chunk" << endl;
	}
     
      // delete active to free it up for the next chunk
      time(&stop);
      predict_time[nproj-proj_no] += difftime(stop,start_predict);
      total_time[nproj-proj_no] += difftime(stop,start_filter);
      free_ActiveDataSet(active);
    }
}




template <typename EigenType>
void evaluate_projection(const std::vector<EigenType*>& x, 
			 const std::vector<SparseMb*>& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 const std::vector<std::string>& setnames,
			 bool allproj, bool verbose, ostream& out)
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

template <typename EigenType>
void evaluate_projection_chunks(const std::vector<EigenType*>& x, 
				const std::vector<SparseMb*>& y, 
				const string& ova_file, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				const std::vector<std::string>& setnames,
				bool allproj, bool verbose, ostream& out)
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



template <typename EigenType>
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

#endif //_EVALUATE_H
