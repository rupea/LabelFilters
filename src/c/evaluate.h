#ifndef _EVALUATE_H
#define _EVALUATE_H

#include "predict.h"
#include "utils.h"

using namespace std;


void output_perfs(double MicroF1, double  MacroF1, double MacroF1_2, double MicroPrecision, double MacroPrecision, double MicroRecall, double MacroRecall, double Top1, double Top5, double Top10, double Prec1, double Prec5, double Prec10, const vector<size_t>& nact, size_t total_preds, double total_time, string str="", ostream& out = cout)
{
  for (int proj =0; proj < nact.size(); proj++)
    {
      out << str <<  "Active_" << proj << "  " << nact[proj]*1.0/total_preds << " (" << nact[proj] << "/" << total_preds << ")" << endl; 
    }
  out << str << "MicroF1  " << MicroF1 << endl;
  out << str << "MacroF1  " << MacroF1 << endl;
  out << str << "MacroF1_2  " << MacroF1_2 << endl;
  out << str << "MicroPrecision  " << MicroPrecision << endl;
  out << str << "MacroPrecision  " << MacroPrecision << endl;
  out << str << "MicroRecall  " << MicroRecall << endl;
  out << str << "MacroRecall  " << MacroRecall << endl;
  out << str << "Top1  " << Top1 << endl;
  out << str << "Prec1  " << Prec1 << endl;
  out << str << "Top5  " << Top5 << endl;
  out << str << "Prec5  " << Prec5 << endl;
  out << str << "Top10  " << Top10 << endl;
  out << str << "Prec10  " << Prec10 << endl;
  out << str << "nactive  " << nact.back() << endl;
  out << str << "prc_active  " << nact.back()*1.0/total_preds << endl;
  out << str << "time  " << total_time << endl;
}


template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 bool verbose, ostream& out,
			 double& MicroF1, double& MacroF1, double& MacroF1_2,
			 double& MicroPrecision, double& MacroPrecision,
			 double& MicroRecall, double& MacroRecall, 
			 double& Top1, double& Top5, double& Top10, 
			 double& Prec1, double& Prec5, double& Prec10,
			 size_t& nact, double& act_prc, double& total_time)
{
  ActiveDataSet* active;
  PredictionSet* predictions;
  vector<size_t> no_active;

  time_t start;
  time_t stop;
  
  time(&start);
  int pred_k = k>10?k:10;
  size_t total_preds = (static_cast<size_t> (y.rows()))*(static_cast<size_t> (y.cols()));
  if (verbose)
    {
      cout << "Predict ... " << endl;
    }
  if (wmat)
    {
      active = getactive (no_active, x, *wmat, *lmat, *umat, projname, verbose);
    }
  else
    {
      active = NULL;
      no_active.push_back(total_preds);
    }
  predictions = predict(x, ovaW, active, nact, verbose, thresh, pred_k); 
  act_prc = nact*1.0/total_preds;
  if (verbose)
    {
      cout << "Done predict" << endl;
    }
  time(&stop);
  total_time = difftime(stop,start);
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }
  predictions->ThreshMetrics(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, y, thresh, k);
  predictions->TopMetrics(Prec1, Top1, Prec5, Top5, Prec10, Top10, y);
  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }
  
  delete predictions;
  if (active)
    {   
      for(ActiveDataSet::iterator actit = active->begin(); actit !=active->end();actit++)
	{
	  delete (*actit);
	}
      delete active;
    }

  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, no_active, total_preds, total_time, projname + "  ", out);
}


template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname, 
			 bool verbose, ostream& out = cout)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;

  evaluate_projection(x, y, ovaW, wmat, lmat, umat, thresh, k, projname, 
		      verbose, out,
		      MicroF1, MacroF1, MacroF1_2,
		      MicroPrecision, MacroPrecision,
		      MicroRecall, MacroRecall, 
		      Top1, Top5, Top10, 
		      Prec1, Prec5, Prec10,
		      nact, act_prc, total_time);
}


template <typename EigenType>
void evaluate_projection_chunks(const EigenType& x, const SparseMb& y, 
				const string& ova_file, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				bool verbose, ostream& out,
				double& MicroF1, double& MacroF1, double& MacroF1_2,
				double& MicroPrecision, double& MacroPrecision,
				double& MicroRecall, double& MacroRecall, 
				double& Top1, double& Top5, double& Top10, 
				double& Prec1, double& Prec5, double& Prec10,
				size_t& nact, double& act_prc, double& total_time)
{
  assert(chunks > 0);
  ActiveDataSet* active = NULL;

  time_t start;
  time_t stop;
  total_time = 0;

  size_t nact_chunk;
  DenseColMf ovaW;
  DenseColM lmat_chunk;
  DenseColM umat_chunk;
  size_t dim = x.cols();
  size_t noClasses = y.cols();
  size_t n = x.rows();
  size_t total_preds = n*noClasses;

  PredictionSet* predictions = new PredictionSet(n);
  vector<size_t> no_active;
  if (wmat)
    {
      no_active.resize(wmat->cols());
    }
  else
    {
      no_active.push_back(total_preds);
    }
  size_t start_class = 0;  
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
      
      time(&start);
      int pred_k = k>10?k:10;
      if (verbose)
	{
	  cout << "Predict chunk ... " << endl;
	}
      if (wmat)
	{
	  // this copies data
	  lmat_chunk = lmat->block(start_class,0,chunk_size,lmat->cols());
	  umat_chunk = umat->block(start_class,0,chunk_size,umat->cols());
	  active = getactive (no_active, x, *wmat, lmat_chunk, umat_chunk, projname, verbose);
	}
      predict(predictions, x, ovaW, active, nact_chunk, verbose, thresh, pred_k, start_class); 
      nact+=nact_chunk; 
      if (verbose)
	{
	  cout << "Done predict chunk" << endl;
	}

      // delete active to free it up for the next chunk
      if (active)
	{   
	  for(ActiveDataSet::iterator actit = active->begin(); actit !=active->end();actit++)
	    {
	      delete (*actit);
	    }
	  delete active;
	}
      
      time(&stop);
      total_time += difftime(stop,start);
      start_class = start_class+chunk_size;
    }
  assert(start_class == noClasses);

  act_prc = nact*1.0/total_preds;
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }
  predictions->ThreshMetrics(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, y, thresh, k);
  predictions->TopMetrics(Prec1, Top1, Prec5, Top5, Prec10, Top10, y);
  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }
  
  delete predictions;

  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, no_active, total_preds, total_time, projname + "  ", out);
}


template <typename EigenType>
void evaluate_projection_chunks(const EigenType& x, const SparseMb& y, 
			       const string& ova_file, int chunks,
			       const DenseColM* wmat, const DenseColM* lmat,
			       const DenseColM* umat,
			       predtype thresh, int k, const string& projname, 
			       bool verbose, ostream& out = cout)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;
  
  evaluate_projection_chunks(x, y, ova_file, chunks, wmat, lmat, umat, thresh, k,
			     projname, verbose, out,
			     MicroF1, MacroF1, MacroF1_2,
			     MicroPrecision, MacroPrecision,
			     MicroRecall, MacroRecall, 
			     Top1, Top5, Top10, 
			     Prec1, Prec5, Prec10,
			     nact, act_prc, total_time);
}




#if 0
template <typename EigenType>
void evaluate_full(const EigenType& x, const SparseMb& y, 
		   const DenseColMf& ovaW,
		   predtype thresh, int k, const string& projname, bool verbose,
		   double& MicroF1, double& MacroF1, double& MacroF1_2,
		   double& MicroPrecision, double& MacroPrecision,
		   double& MicroRecall, double& MacroRecall, 
		   double& Top1, double& Top5, double& Top10, 
		   double& Prec1, double& Prec5, double& Prec10,
		   size_t& nact, double& act_prc, double& total_time)
{
  PredictionSet* predictions;
  nact= (static_cast<size_t> (y.rows()))*(static_cast<size_t> (y.cols()));
  vector<size_t> no_active;
  no_active.push_back(nact);
  act_prc = 1.0;
  time_t start;
  time_t stop;
  
  time(&start);
  if (verbose)
    {
      cout << "Predict full ... " << endl;
    }
  size_t foo;
  int pred_k = k>10?k:10;
  predictions = predict(x, ovaW, NULL, foo, verbose, thresh, pred_k); 
  if (verbose)
    {
      cout << "Done predict full." << endl;
    }
  time(&stop);
  total_time = difftime(stop,start);
  if (verbose)
    {
      cout << "Evaluate full... " << endl;
    }
  predictions->ThreshMetrics(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, y, thresh, k);
  predictions->TopMetrics(Prec1, Top1, Prec5, Top5, Prec10, Top10, y);
  if (verbose)
    {
      cout << "Done evaluate full." << endl;
    }
  
  delete predictions;
  
  output_perfs( MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, no_active, nact, total_time, "full  ");
}

template <typename EigenType>
void evaluate_full(const EigenType& x, const SparseMb& y, 
		   const DenseColMf& ovaW,
		   predtype thresh, int k, const string& projname, bool verbose)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;
  evaluate_full(x, y, ovaW, thresh, k, projname, verbose,
		MicroF1, MacroF1, MacroF1_2,
		MicroPrecision, MacroPrecision,
		MicroRecall, MacroRecall, 
		Top1, Top5, Top10, 
		Prec1, Prec5, Prec10,
		nact, act_prc, total_time);
}
#endif

#endif //_EVALUATE_H
