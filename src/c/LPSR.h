#ifndef _LPSR_H
#define _LPSR_H

#include "predict.h"
#include "evaluate.h"
#include "utils.h"
#include "EigenIO.h"
#include <boost/dynamic_bitset.hpp>


inline ActiveDataSet* counting_heuristic(const SparseMb& y, const Eigen::VectorXi& assignments, int k, int C, bool verbose)
{
  using boost::dynamic_bitset;

  ActiveDataSet* active = new ActiveDataSet(k);
  size_t n = y.rows();
  int noClasses = y.cols();
  DenseM counts(k, noClasses);
  counts.setZero();
  if (C > noClasses)
    {
      C = noClasses;
    }
  cout << "C " << C << " k " << k << " noclasses " << noClasses << " n " << n <<endl;
  cout << "Assignments size " << assignments.size() << endl;
  cout << "assignments max " << assignments.maxCoeff();
  cout << " assignments min " << assignments.minCoeff() << endl;
  for (size_t i=0; i<n; i++)
    {
      for( SparseMb::InnerIterator it(y,i); it; ++it)
	{
	  if (it.value())
	    {
	      counts(assignments(i),it.col())++;
	    }
	}
    }
  cout << "got to here" << endl;
  // break ties arbitrarily
  counts += (DenseM::Random(k,noClasses)/1e-2);
  // sort the columns
  std::vector<int> cranks(noClasses);
  cout << "start sorting" << endl;
  for (int j=0;j<k;j++)
    {
      active->at(j) = new dynamic_bitset<>(noClasses); 
      sort_index(counts.row(j),cranks);
      for (int l=noClasses-1;l>=noClasses-C;l--)
	{
	  (*active)[j]->set(cranks[l]);
	}
    }  
  return active;
}

template <typename Eigentype> inline
ActiveDataSet* getactive_LPSR_counting(size_t& nact, const Eigentype& x, const DenseColM& centers, const ActiveDataSet& active_classes, bool spherical, bool verbose = false)
{
  size_t n = x.rows();
  ActiveDataSet* active = new ActiveDataSet(n);
  Eigen::VectorXi assignments(n);
  
  cluster_test_kmeans(assignments,centers,x,spherical,verbose);
  nact = 0;
  for(size_t i=0;i<n;i++)
    {
      (*active)[i] = active_classes[assignments(i)];
      nact += (*active)[i]->count();
    }

  return active;
}


template <typename EigenType> inline
void evaluate_LPSR_chunks(const EigenType& x, const SparseMb& y, 
			  const string& ova_file, int chunks,
			  const DenseColM& centers, const ActiveDataSet& active_classes,
			  predtype thresh, int k, const string& evalname,
			  bool validation, bool spherical, bool verbose, ostream& out,
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
  using boost::dynamic_bitset;
  assert(chunks > 0);
  double time_chunk;
  nact_final = 0;
  size_t nact_valid=0;
  time_t start;
  time_t stop;
  size_t nact_chunk=0,nact_valid_chunk=0;
  size_t dim = x.cols();
  size_t noClasses = y.cols();
  size_t n = x.rows();
  PredictionSet* predictions;
  PredictionSet* predictions_valid;


  std::vector<double> MicroF1_valid(1), MacroF1_valid(1);
  std::vector<double> MacroF1_2_valid(1);
  std::vector<double> MicroPrecision_valid(1);
  std::vector<double> MacroPrecision_valid(1);
  std::vector<double> MicroRecall_valid(1), MacroRecall_valid(1);
  std::vector<double> Top1_valid(1), Top5_valid(1);
  std::vector<double> Top10_valid(1);
  std::vector<double> Prec1_valid(1), Prec5_valid(1);
  std::vector<double> Prec10_valid(1);
  std::vector<double> total_time_valid(1);
  std::vector<double> filter_time_valid(1);
  std::vector<double> predict_time_valid(1);

  std::vector<double> MicroF1(1), MacroF1(1);
  std::vector<double> MacroF1_2(1);
  std::vector<double> MicroPrecision(1), MacroPrecision(1);
  std::vector<double> MicroRecall(1), MacroRecall(1);
  std::vector<double> Top1(1), Top5(1), Top10(1);
  std::vector<double> Prec1(1), Prec5(1), Prec10(1);
  std::vector<double> total_time(1);
  std::vector<double> filter_time(1);
  std::vector<double> predict_time(1);

  size_t n_valid=0;
  size_t total_preds_valid = 0;
  if (validation)
    {
      n_valid = n/2;
      total_preds_valid = n_valid*noClasses;
      predictions_valid = new PredictionSet(n_valid);
      total_time_valid[0] = 0;
      n -= n_valid;
    }
  predictions = new PredictionSet(n);
  total_time[0] = 0;
 
  ActiveDataSet* active_chunk_valid=NULL;
  if (validation)
    {
      active_chunk_valid = new ActiveDataSet(n_valid);
    }
  ActiveDataSet* active_chunk = new ActiveDataSet(n);

  VectorXsz no_active_valid(1);
  VectorXsz no_active(1);  

  size_t total_preds = n*noClasses;
    
  size_t start_class = 0;  

  time(&start);
  ActiveDataSet* active_ptr = getactive_LPSR_counting(nact_final, x, centers, active_classes, spherical, verbose);
  nact_final = 0;
  time(&stop);
  if (validation)
    {
      filter_time_valid[0]=filter_time[0]=difftime(stop,start)*1.0/2;
    }
  else
    {
      filter_time[0]=difftime(stop,start)*1.0;
    }
      

  { // have an internal block so that ovaW goes out of scope at the end of it 
    // and memory is released
    DenseColMf ovaW;
    ActiveDataSet active = *active_ptr;    
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
       	
	if (validation)
	  {
	    for (size_t i = 0;i<n_valid; i++)
	      {
		// dynamic_bitset assumes that the most significatnt bit is the last one, so the
		// shifting is reversed from what one would normally think
                boost::dynamic_bitset<> a = *(active[i]) >> start_class;
		a.resize(chunk_size);
		active_chunk_valid->at(i) = new dynamic_bitset<>(a);
		//active_chunk_valid[i]->resize(chunk_size);
	      }	    
	    time(&start);
	    predict(predictions_valid, x.topRows(n_valid), ovaW, active_chunk_valid, nact_valid_chunk,
		    verbose, thresh, k, start_class);
	    time(&stop);	 
	    predict_time_valid[0] += difftime(start,stop);
	    nact_valid+=nact_valid_chunk;
	    free_ActiveDataSet(active_chunk_valid);
	  }	    
	// if no vaidation is performed then n is the number of instances
	for (size_t i = 0;i<n; i++)
	  {
	    // dynamic_bitset assumes that the most significatnt bit is the last one, so the
	    // shifting is reversed from what one would normally think
	    dynamic_bitset<> a = *(active[i + n_valid]) >> start_class;
	    a.resize(chunk_size);
	    active_chunk->at(i) = new dynamic_bitset<>(a);
	    //active_chunk[i-n_valid].resize(chunk_size);
	  }
	time(&start);
	predict(predictions, x.bottomRows(n), ovaW, active_chunk, nact_chunk,
		verbose, thresh, k, start_class);
	time(&stop);
	predict_time[0] += difftime(stop,start);
	nact_final += nact_chunk;
	free_ActiveDataSet(active_chunk);
	start_class = start_class + chunk_size;
      }
  } // we don't need ovaW any more
  if (active_chunk)
    {
      delete active_chunk;
    }
  if (active_chunk_valid)
    {
      delete active_chunk_valid;
    }
  if (active_ptr)
    {
      // Do not delete the contents of the AcitveDataSet since they just point to the entries in 
      // active_classes. These will get deleted when active_classes get deleted. 
      delete active_ptr;
    }
  
  assert(start_class == noClasses);
  if (validation)
    {
      total_time_valid[0] = filter_time_valid[0]+predict_time_valid[0];
      no_active_valid[0] = nact_valid;
    }
  total_time[0] = filter_time[0]+predict_time[0];
  no_active[0] = nact_final;

  act_prc_final = nact_final*1.0/total_preds;
  if (verbose)
    {
      cout << "Evaluate... " << endl;
    }
  
  if (validation)
    {      
      predictions_valid->ThreshMetrics(MicroF1_valid[0], MacroF1_valid[0], MacroF1_2_valid[0], MicroPrecision_valid[0], MacroPrecision_valid[0], MicroRecall_valid[0], MacroRecall_valid[0], y.topRows(n_valid), thresh, k);
      predictions_valid->TopMetrics(Prec1_valid[0], Top1_valid[0], Prec5_valid[0], Top5_valid[0], Prec10_valid[0], Top10_valid[0], y.topRows(n_valid));
      delete predictions_valid;
    }
  predictions->ThreshMetrics(MicroF1[0], MacroF1[0], MacroF1_2[0], MicroPrecision[0], MacroPrecision[0], MicroRecall[0], MacroRecall[0], y.bottomRows(n), thresh, k);
  predictions->TopMetrics(Prec1[0], Top1[0], Prec5[0], Top5[0], Prec10[0], Top10[0], y.bottomRows(n));
  delete predictions;
  
  if (validation)
    {
      output_perfs(MicroF1_valid, MacroF1_valid, MacroF1_2_valid, 
		   MicroPrecision_valid, MacroPrecision_valid, MicroRecall_valid, MacroRecall_valid, 
		   Top1_valid, Top5_valid, Top10_valid, Prec1_valid, Prec5_valid, Prec10_valid, 
		   no_active_valid, total_preds_valid, total_time_valid, filter_time_valid, predict_time_valid, evalname + "  valid_", out);
    }
  output_perfs(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, 
	       MicroRecall, MacroRecall, 
	       Top1, Top5, Top10, Prec1, Prec5, Prec10, 
	       no_active, total_preds, total_time, filter_time, predict_time, evalname + "  ", out);
  
  
  if (verbose)
    {
      cout << "Done evaluate." << endl;
    }

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
  
}



template <typename EigenType> inline
void evaluate_LPSR_chunks(const EigenType& x, const SparseMb& y, 
			  const string& ova_file, int chunks,
			  const DenseColM& centers, const ActiveDataSet& active_classes,
			  predtype thresh, int k, const string& evalname,
			  bool validation, bool spherical, bool verbose, ostream& out)
{
  size_t nact;
  double total_time, act_prc;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall;
  double Top1, Top5, Top10, Prec1, Prec5, Prec10;
  evaluate_LPSR_chunks(x, y, ova_file, chunks, centers, active_classes,
		       thresh, k, evalname, validation, spherical, verbose, out,
		       MicroF1, MacroF1, MacroF1_2,
		       MicroPrecision, MacroPrecision,
		       MicroRecall, MacroRecall, 
		       Top1, Top5, Top10, 
		       Prec1, Prec5, Prec10,
		       nact, act_prc, total_time);
}



template <typename EigenType> inline
void train_LPSR(DenseColM& centers, ActiveDataSet*& active_classes, 
		const EigenType& x, const SparseMb& y, int C, int iterations, bool spherical, bool verbose=false)
{
  size_t n = x.rows();
  int k = centers.cols();
  Eigen::VectorXi assignments(n);  
  run_kmeans(centers,assignments,x,iterations,spherical,verbose);
  if (verbose)
    {
      cout << "Done K-means" << endl;
    }
  active_classes = counting_heuristic(y,assignments,k,C,verbose);
  if (verbose)
    {
      cout << "Done counting heuristic" << endl;
    }
}
 
#endif
