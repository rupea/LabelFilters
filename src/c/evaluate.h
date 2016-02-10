#ifndef _EVALUATE_H
#define _EVALUATE_H

#include "predict.h"

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
		  string str="", ostream& out = cout);

// -------- inline template declarations --------

template <typename EigenType>
void predict_chunk(PredictionSet* predictions, size_t& nact,
		   const EigenType& x, const DenseColMf& ovaW,
		   const size_t start_class, const ActiveDataSet* active,
		   predtype thresh, int k, bool verbose);
template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y,
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 bool validation, bool allproj, bool verbose, ostream& out,
			 double& MicroF1_final, double& MacroF1_final,
			 double& MacroF1_2_final,
			 double& MicroPrecision_final, double& MacroPrecision_final,
			 double& MicroRecall_final, double& MacroRecall_final,
			 double& Top1_final, double& Top5_final, double& Top10_final,
			 double& Prec1_final, double& Prec5_final,
			 double& Prec10_final,
			 size_t& nact_final, double& act_prc_final,
			 double& total_time_final);
template <typename EigenType>
void get_projection_measures(const EigenType& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue);
template <typename EigenType>
void evaluate_projection(const EigenType& x, const SparseMb& y,
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 bool validation, bool allproj, bool verbose, ostream& out = cout);
template <typename EigenType>
void evaluate_projection_chunks(const EigenType& x, const SparseMb& y,
				const string& ova_file, size_t chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				bool validation, bool allproj, bool verbose, ostream& out = cout);
template <typename EigenType>
void evaluate_projection_chunks(const EigenType& x, const SparseMb& y,
				const string& ova_file, size_t chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				bool validation, bool allproj, bool verbose, ostream& out,
				double& MicroF1_final, double& MacroF1_final,
				double& MacroF1_2_final,
				double& MicroPrecision_final, double& MacroPrecision_final,
				double& MicroRecall_final, double& MacroRecall_final,
				double& Top1_final, double& Top5_final, double& Top10_final,
				double& Prec1_final, double& Prec5_final,
				double& Prec10_final,
				size_t& nact_final, double& act_prc_final,
				double& total_time_final);

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

  output_perfs( MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, no_active, nact, total_time, total_time, total_time, "full  ");
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


// -------- inline template definitions --------> evaluate.hh (not needed if linking with library)

#endif //_EVALUATE_H
