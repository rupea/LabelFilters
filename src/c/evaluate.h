#ifndef _EVALUATE_H
#define _EVALUATE_H

#include "predict.h"
#include "utils.h"




void output_perfs(double MicroF1, double  MacroF1, double MacroF1_2, double MicroPrecision, double MacroPrecision, double MicroRecall, double MacroRecall, double Top1, double Top5, double Top10, double Prec1, double Prec5, double Prec10, size_t nact, double act_prc, double total_time, string str="")
{
  cout << str << "MicroF1  " << MicroF1 << endl;
  cout << str << "MacroF1  " << MacroF1 << endl;
  cout << str << "MacroF1_2  " << MacroF1_2 << endl;
  cout << str << "MicroPrecision  " << MicroPrecision << endl;
  cout << str << "MacroPrecision  " << MacroPrecision << endl;
  cout << str << "MicroRecall  " << MicroRecall << endl;
  cout << str << "MacroRecall  " << MacroRecall << endl;
  cout << str << "Top1  " << Top1 << endl;
  cout << str << "Prec1  " << Prec1 << endl;
  cout << str << "Top5  " << Top5 << endl;
  cout << str << "Prec5  " << Prec5 << endl;
  cout << str << "Top10  " << Top10 << endl;
  cout << str << "Prec10  " << Prec10 << endl;
  cout << str << "nactive  " << nact << endl;
  cout << str << "prc_active  " << act_prc << endl;
  cout << str << "time  " << total_time << endl;
}



template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y, 
			 const DenseColMf& ovaW,
			 const DenseColM& wmat, const DenseColM& lmat,
			 const DenseColM& umat,
			 predtype thresh, int k, const string& projname,
			 bool verbose,
			 double& MicroF1, double& MacroF1, double& MacroF1_2,
			 double& MicroPrecision, double& MacroPrecision,
			 double& MicroRecall, double& MacroRecall, 
			 double& Top1, double& Top5, double& Top10, 
			 double& Prec1, double& Prec5, double& Prec10,
			 size_t& nact, double& act_prc, double& total_time)
{
  ActiveDataSet* active;
  PredictionSet* predictions;
  
  // size_t nact;
  time_t start;
  time_t stop;
  //  double total_time;
  //  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;
  
  time(&start);
  if (verbose)
    {
      cout << "Predict projection ... " << endl;
    }
  active = getactive (x, wmat, lmat, umat, projname, verbose);
  predictions = predict(x, ovaW, active, nact, verbose, thresh, 10); 
  size_t total_preds = (static_cast<size_t> (y.rows()))*(static_cast<size_t> (y.cols()));
  act_prc = nact*1.0/total_preds;
  if (verbose)
    {
      cout << "Done predict projection" << endl;
    }
  time(&stop);
  total_time = difftime(stop,start);
  if (verbose)
    {
      cout << "Evaluate projection... " << endl;
    }
  predictions->ThreshMetrics(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, y, thresh, k);
  predictions->TopMetrics(Prec1, Top1, Prec5, Top5, Prec10, Top10, y);
  if (verbose)
    {
      cout << "Done evaluate projection." << endl;
    }
  
  delete predictions;
  for(ActiveDataSet::iterator actit = active->begin(); actit !=active->end();actit++)
    {
      delete (*actit);
    }
  delete active;

  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, nact, act_prc, total_time, projname + "  ");
}


template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y, 
			 const DenseColMf& ovaW,
			 const DenseColM& wmat, const DenseColM& lmat,
			 const DenseColM& umat,
			 predtype thresh, int k, const string& projname, 
			 bool verbose)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;

  evaluate_projection(x, y, ovaW, wmat, lmat, umat, thresh, k, projname, 
		      verbose,
		      MicroF1, MacroF1, MacroF1_2,
		      MicroPrecision, MacroPrecision,
		      MicroRecall, MacroRecall, 
		      Top1, Top5, Top10, 
		      Prec1, Prec5, Prec10,
		      nact, act_prc, total_time);
}


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
  act_prc = 1.0;
  time_t start;
  time_t stop;
  
  time(&start);
  if (verbose)
    {
      cout << "Predict full ... " << endl;
    }
  size_t foo;
  predictions = predict(x, ovaW, NULL, foo, verbose, thresh, 10); 
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
  
  output_perfs( MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, nact, act_prc, total_time, "full  ");
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


#endif //_EVALUATE_H
