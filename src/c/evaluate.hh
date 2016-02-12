#ifndef EVALUATE_HH
#define EVALUATE_HH
/** \file
 * inline template (not needed for library linking - libmcfilter includes DenseM & SparseM support)
 */

#include "evaluate.h"
#include "predict.hh"   // oh, actually need the inline defn of predict for library
#include "utils.h"
#include "EigenIO.h"

template <typename EigenType> inline
void predict_chunk(PredictionSet* predictions, size_t& nact,
		   const EigenType& x, const DenseColMf& ovaW,
		   const size_t start_class, const ActiveDataSet* active,
		   predtype thresh, int k, bool verbose)
{
  size_t pred_k = (k>10U?static_cast<size_t>(k):size_t{10U});
  //size_t n = x.rows();
  if (verbose)
    {
      cout << "Predict chunk ... " << endl;
    }
  predict(predictions, x, ovaW, active, nact, verbose, thresh, pred_k, start_class);

  if (verbose)
    {
      cout << "Done predict chunk" << endl;
    }
}


template <typename EigenType> inline
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
			 double& total_time_final)
{

  time_t start;
  time_t stop;
  size_t n = x.rows();
  //size_t dim = x.cols();
  size_t noClasses = y.cols();
  int nproj = wmat?wmat->cols():1;
  ActiveDataSet* active=NULL;
  int proj_no;

  std::vector<PredictionSet*> predictions(allproj?nproj:1);
  std::vector<PredictionSet*> predictions_valid(allproj?nproj:1);

  std::vector<double> MicroF1_valid(allproj?nproj:1), MacroF1_valid(allproj?nproj:1);
  std::vector<double> MacroF1_2_valid(allproj?nproj:1);
  std::vector<double> MicroPrecision_valid(allproj?nproj:1);
  std::vector<double> MacroPrecision_valid(allproj?nproj:1);
  std::vector<double> MicroRecall_valid(allproj?nproj:1), MacroRecall_valid(allproj?nproj:1);
  std::vector<double> Top1_valid(allproj?nproj:1), Top5_valid(allproj?nproj:1);
  std::vector<double> Top10_valid(allproj?nproj:1);
  std::vector<double> Prec1_valid(allproj?nproj:1), Prec5_valid(allproj?nproj:1);
  std::vector<double> Prec10_valid(allproj?nproj:1);
  std::vector<double> total_time_valid(allproj?nproj:1);

  std::vector<double> MicroF1(allproj?nproj:1), MacroF1(allproj?nproj:1);
  std::vector<double> MacroF1_2(allproj?nproj:1);
  std::vector<double> MicroPrecision(allproj?nproj:1), MacroPrecision(allproj?nproj:1);
  std::vector<double> MicroRecall(allproj?nproj:1), MacroRecall(allproj?nproj:1);
  std::vector<double> Top1(allproj?nproj:1), Top5(allproj?nproj:1), Top10(allproj?nproj:1);
  std::vector<double> Prec1(allproj?nproj:1), Prec5(allproj?nproj:1), Prec10(allproj?nproj:1);
  std::vector<double> total_time(allproj?nproj:1);

  VectorXsz no_active_valid(nproj);
  VectorXsz no_active(nproj);

  size_t nact_valid = 0;
  size_t n_valid=0;
  size_t total_preds_valid = 0;
  if (validation)
    {
      n_valid = n/2;
      total_preds_valid = n_valid*noClasses;
      for (proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  predictions_valid[nproj-proj_no] = new PredictionSet(n_valid);
	}
      n -= n_valid;
    }
  for (proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
    {
      predictions[nproj-proj_no] = new PredictionSet(n);
    }
  size_t total_preds = n*noClasses;


  if (!wmat)
    {
      if (validation)
	{
	  no_active_valid[0] = total_preds_valid;
	}
      no_active[0] = total_preds;
    }

  for (proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
    {
      DenseColM wmat_truncated;
      if (wmat)
	{
	  wmat_truncated = wmat->leftCols(proj_no);
	}
      if (validation) // predictions using 'top half' of training examples
	{
	  time(&start);
	  if (wmat)
	    {
	      active = getactive (no_active_valid, x.topRows(n_valid), wmat_truncated, *lmat, *umat, verbose);
	    }
	  predict_chunk(predictions_valid[nproj-proj_no], nact_valid, x.topRows(n_valid), ovaW, 0, active,
			thresh, k, verbose);
	  // delete active to free it up for the next chunk
	  free_ActiveDataSet(active); free(active); active=NULL;
	  time(&stop);
	  total_time_valid[nproj-proj_no] += difftime(stop,start);
	}
      // if no vaidation is performed then n is the number of instances
      // predict using all (or some 'bottom half' of training examples)
      time(&start);
      if (wmat)
	{
	  active = getactive (no_active, x.bottomRows(n), wmat_truncated, *lmat, *umat, verbose);
	}
      predict_chunk(predictions[nproj-proj_no], nact_final,
		    x.bottomRows(n), ovaW, 0, active,
		    thresh, k, verbose);
      // delete active to free it up for the next chunk
      free_ActiveDataSet(active); free(active); active=NULL;
      time(&stop);
      total_time[nproj-proj_no] += difftime(stop,start);
    }
  act_prc_final = nact_final*1.0/total_preds;
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }

  for (proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
    {
      if (validation)
	{
	  predictions_valid[nproj-proj_no]->ThreshMetrics(MicroF1_valid[nproj-proj_no], MacroF1_valid[nproj-proj_no], MacroF1_2_valid[nproj-proj_no], MicroPrecision_valid[nproj-proj_no], MacroPrecision_valid[nproj-proj_no], MicroRecall_valid[nproj-proj_no], MacroRecall_valid[nproj-proj_no], y.topRows(n_valid), thresh, k);
	  predictions_valid[nproj-proj_no]->TopMetrics(Prec1_valid[nproj-proj_no], Top1_valid[nproj-proj_no], Prec5_valid[nproj-proj_no], Top5_valid[nproj-proj_no], Prec10_valid[nproj-proj_no], Top10_valid[nproj-proj_no], y.topRows(n_valid));
	  delete predictions_valid[nproj-proj_no];
	}
      predictions[nproj-proj_no]->ThreshMetrics(MicroF1[nproj-proj_no], MacroF1[nproj-proj_no], MacroF1_2[nproj-proj_no], MicroPrecision[nproj-proj_no], MacroPrecision[nproj-proj_no], MicroRecall[nproj-proj_no], MacroRecall[nproj-proj_no], y.bottomRows(n), thresh, k);
      predictions[nproj-proj_no]->TopMetrics(Prec1[nproj-proj_no], Top1[nproj-proj_no], Prec5[nproj-proj_no], Top5[nproj-proj_no], Prec10[nproj-proj_no], Top10[nproj-proj_no], y.bottomRows(n));
      delete predictions[nproj-proj_no];
    }

  if (validation)
    {
      output_perfs(MicroF1_valid, MacroF1_valid, MacroF1_2_valid, MicroPrecision_valid, MacroPrecision_valid, MicroRecall_valid, MacroRecall_valid,
		   Top1_valid, Top5_valid, Top10_valid, Prec1_valid, Prec5_valid, Prec10_valid,
		   no_active_valid, total_preds_valid, total_time_valid,total_time_valid,total_time_valid, projname + "  valid_", out);
    }
  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision,
	       MicroRecall, MacroRecall,
	       Top1, Top5, Top10, Prec1, Prec5, Prec10,
	       no_active, total_preds, total_time, total_time, total_time, projname + "  ", out);

  MicroF1_final = MicroF1[0];
  MacroF1_final = MacroF1[0];
  MacroF1_2_final = MacroF1_2[0];
  MicroPrecision_final = MicroPrecision[0];
  MacroPrecision_final =  MacroPrecision[0];
  MicroRecall_final = MicroRecall[0];
  MacroRecall_final = MacroRecall[0];
  Top1_final = Top1[0];
  Top5_final = Top5[0];
  Top10_final = Top10[0];
  Prec1_final = Prec1[0];
  Prec5_final = Prec5[0];
  Prec10_final = Prec10[0];
  total_time_final = total_time[0];

  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }
}


template <typename EigenType> inline
void evaluate_projection(const EigenType& x, const SparseMb& y,
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 bool validation, bool allproj, bool verbose, ostream& out = cout)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall;
  double Top1, Top5, Top10, Prec1, Prec5, Prec10;

  evaluate_projection(x, y, ovaW, wmat, lmat, umat, thresh, k, projname,
		      validation, allproj, verbose, out,
		      MicroF1, MacroF1, MacroF1_2,
		      MicroPrecision, MacroPrecision,
		      MicroRecall, MacroRecall,
		      Top1, Top5, Top10,
		      Prec1, Prec5, Prec10,
		      nact, act_prc, total_time);
}


template <typename EigenType> inline
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
				double& total_time_final)
{
  assert(chunks > 0);
  //double time_chunk;
  nact_final = 0;

  time_t start_filter;
  time_t start_predict;
  time_t stop;
  size_t nact_chunk=0;
  size_t dim = x.cols();
  size_t noClasses = y.cols();
  size_t n = x.rows();
  int nproj = wmat?wmat->cols():1;
  int proj_no;
  ActiveDataSet* active=NULL;


  std::vector<PredictionSet*> predictions(allproj?nproj:1);
  std::vector<PredictionSet*> predictions_valid(allproj?nproj:1);


  std::vector<double> MicroF1_valid(allproj?nproj:1), MacroF1_valid(allproj?nproj:1);
  std::vector<double> MacroF1_2_valid(allproj?nproj:1);
  std::vector<double> MicroPrecision_valid(allproj?nproj:1);
  std::vector<double> MacroPrecision_valid(allproj?nproj:1);
  std::vector<double> MicroRecall_valid(allproj?nproj:1), MacroRecall_valid(allproj?nproj:1);
  std::vector<double> Top1_valid(allproj?nproj:1), Top5_valid(allproj?nproj:1);
  std::vector<double> Top10_valid(allproj?nproj:1);
  std::vector<double> Prec1_valid(allproj?nproj:1), Prec5_valid(allproj?nproj:1);
  std::vector<double> Prec10_valid(allproj?nproj:1);
  std::vector<double> total_time_valid(allproj?nproj:1);
  std::vector<double> filter_time_valid(allproj?nproj:1);
  std::vector<double> predict_time_valid(allproj?nproj:1);

  std::vector<double> MicroF1(allproj?nproj:1), MacroF1(allproj?nproj:1);
  std::vector<double> MacroF1_2(allproj?nproj:1);
  std::vector<double> MicroPrecision(allproj?nproj:1), MacroPrecision(allproj?nproj:1);
  std::vector<double> MicroRecall(allproj?nproj:1), MacroRecall(allproj?nproj:1);
  std::vector<double> Top1(allproj?nproj:1), Top5(allproj?nproj:1), Top10(allproj?nproj:1);
  std::vector<double> Prec1(allproj?nproj:1), Prec5(allproj?nproj:1), Prec10(allproj?nproj:1);
  std::vector<double> total_time(allproj?nproj:1);
  std::vector<double> filter_time(allproj?nproj:1);
  std::vector<double> predict_time(allproj?nproj:1);


  size_t n_valid=0;
  size_t total_preds_valid = 0;
  if (validation)
    {
      n_valid = n/2;
      total_preds_valid = n_valid*noClasses;
      for (proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
	{
	  predictions_valid[nproj-proj_no] = new PredictionSet(n_valid);
	  total_time_valid[nproj-proj_no] = 0;
	  filter_time_valid[nproj-proj_no] = 0;
	  predict_time_valid[nproj-proj_no] = 0;
	}
      n -= n_valid;
    }
  for (proj_no=allproj?1:nproj;proj_no<=nproj;proj_no++)
    {
      predictions[nproj-proj_no] = new PredictionSet(n);
      total_time[nproj-proj_no] = 0;
      filter_time[nproj-proj_no] = 0;
      predict_time[nproj-proj_no] = 0;
    }

  size_t total_preds = n*noClasses;


  VectorXsz no_active_valid(nproj);
  VectorXsz no_active(nproj);
  no_active_valid.setZero();
  no_active.setZero();
  if (!wmat)
    {
      if (validation)
	{
	  no_active_valid[0] = total_preds_valid;
	}
      no_active[0] = total_preds;
    }


  size_t start_class = 0;

  { // have an internal block so that ovaW goes out of scope at the end of it
    // and memory is released
    DenseColMf ovaW;
    DenseColM lmat_chunk;
    DenseColM umat_chunk;

    for (size_t chunk = 0U; chunk < chunks; ++chunk)
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

	for (proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
	  {
	    DenseColM wmat_truncated;
	    if (wmat)
	      {
		wmat_truncated = wmat->leftCols(proj_no);
	      }
	    if (validation)
	      {
		VectorXsz no_active_valid_chunk;
		time(&start_filter);
		if (wmat)
		  {
		    active = getactive (no_active_valid_chunk, x.topRows(n_valid), wmat_truncated, lmat_chunk, umat_chunk, verbose);
		  }
		time(&start_predict);
		filter_time_valid[nproj-proj_no] += difftime(start_predict,start_filter);
		predict_chunk(predictions_valid[nproj-proj_no], nact_chunk,
			      x.topRows(n_valid), ovaW, start_class, active,
			      thresh, k, verbose);
		// delete active to free it up for the next chunk
		free_ActiveDataSet(active);
		time(&stop);
		predict_time_valid[nproj-proj_no] += difftime(stop,start_predict);
		total_time_valid[nproj-proj_no] += difftime(stop,start_filter);
		if (proj_no == nproj)
		  {
		    no_active_valid += no_active_valid_chunk;
		  }
	      }
	    // if no vaidation is performed then n is the number of instances
	    VectorXsz no_active_chunk;
	    time(&start_filter);
	    if (wmat)
	      {
		active = getactive (no_active_chunk, x.bottomRows(n), wmat_truncated, lmat_chunk, umat_chunk, verbose);
	      }
	    time(&start_predict);
	    filter_time[nproj-proj_no] += difftime(start_predict,start_filter);
	    predict_chunk(predictions[nproj-proj_no], nact_chunk,
			  x.bottomRows(n), ovaW, start_class, active,
			  thresh, k, verbose);
	    // delete active to free it up for the next chunk
	    free_ActiveDataSet(active);
	    time(&stop);
	    predict_time[nproj-proj_no] += difftime(stop,start_predict);
	    total_time[nproj-proj_no] += difftime(stop,start_filter);
	    if (proj_no == nproj)
	      {
		nact_final += nact_chunk;
		no_active += no_active_chunk;
	      }
	  }
	start_class = start_class+chunk_size;
      }
  } // we don't need ovaW any more
  assert(start_class == noClasses);

  act_prc_final = nact_final*1.0/total_preds;
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }

  for (proj_no=allproj?1:nproj; proj_no <= nproj; proj_no++)
    {
      if (validation)
	{
	  predictions_valid[nproj-proj_no]->ThreshMetrics(MicroF1_valid[nproj-proj_no], MacroF1_valid[nproj-proj_no], MacroF1_2_valid[nproj-proj_no], MicroPrecision_valid[nproj-proj_no], MacroPrecision_valid[nproj-proj_no], MicroRecall_valid[nproj-proj_no], MacroRecall_valid[nproj-proj_no], y.topRows(n_valid), thresh, k);
	  predictions_valid[nproj-proj_no]->TopMetrics(Prec1_valid[nproj-proj_no], Top1_valid[nproj-proj_no], Prec5_valid[nproj-proj_no], Top5_valid[nproj-proj_no], Prec10_valid[nproj-proj_no], Top10_valid[nproj-proj_no], y.topRows(n_valid));
	  delete predictions_valid[nproj-proj_no];
	}
      predictions[nproj-proj_no]->ThreshMetrics(MicroF1[nproj-proj_no], MacroF1[nproj-proj_no], MacroF1_2[nproj-proj_no], MicroPrecision[nproj-proj_no], MacroPrecision[nproj-proj_no], MicroRecall[nproj-proj_no], MacroRecall[nproj-proj_no], y.bottomRows(n), thresh, k);
      predictions[nproj-proj_no]->TopMetrics(Prec1[nproj-proj_no], Top1[nproj-proj_no], Prec5[nproj-proj_no], Top5[nproj-proj_no], Prec10[nproj-proj_no], Top10[nproj-proj_no], y.bottomRows(n));
      delete predictions[nproj-proj_no];
    }

  if (validation)
    {
      output_perfs(MicroF1_valid, MacroF1_valid, MacroF1_2_valid, MicroPrecision_valid, MacroPrecision_valid, MicroRecall_valid, MacroRecall_valid,
		   Top1_valid, Top5_valid, Top10_valid, Prec1_valid, Prec5_valid, Prec10_valid,
		   no_active_valid, total_preds_valid, total_time_valid, filter_time_valid, predict_time_valid, projname + "  valid_", out);
    }
  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision,
	       MicroRecall, MacroRecall,
	       Top1, Top5, Top10, Prec1, Prec5, Prec10,
	       no_active, total_preds, total_time,filter_time,predict_time, projname + "  ", out);

  MicroF1_final = MicroF1[0];
  MacroF1_final = MacroF1[0];
  MacroF1_2_final = MacroF1_2[0];
  MicroPrecision_final = MicroPrecision[0];
  MacroPrecision_final =  MacroPrecision[0];
  MicroRecall_final = MicroRecall[0];
  MacroRecall_final = MacroRecall[0];
  Top1_final = Top1[0];
  Top5_final = Top5[0];
  Top10_final = Top10[0];
  Prec1_final = Prec1[0];
  Prec5_final = Prec5[0];
  Prec10_final = Prec10[0];
  total_time_final = total_time[0];

  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }

}


template <typename EigenType> inline
void evaluate_projection_chunks(const EigenType& x, const SparseMb& y,
				const string& ova_file, size_t chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				bool validation, bool allproj, bool verbose, ostream& out = cout)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall;
  double Top1, Top5, Top10, Prec1, Prec5, Prec10;

  evaluate_projection_chunks(x, y, ova_file, chunks, wmat, lmat, umat, thresh, k,
			     projname, validation, allproj, verbose, out,
			     MicroF1, MacroF1, MacroF1_2,
			     MicroPrecision, MacroPrecision,
			     MicroRecall, MacroRecall,
			     Top1, Top5, Top10,
			     Prec1, Prec5, Prec10,
			     nact, act_prc, total_time);
}




template <typename EigenType> inline
void get_projection_measures(const EigenType& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue)
{
  size_t n = x.rows();
  //size_t dim = x.cols();
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
