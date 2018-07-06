#ifndef MCLINEARCLASS_H
#define MCLINEARCLASS_H
/** \file
 * Class to make and evaluate predictions of linear classifiers with label filters applied
 */

#include "mcprojector.h"
#include "PredictionSet.h"
#include <memory>

class MCxyData;
class linearModel;

struct PerfStruct;

class MClinearClassifier : public MCprojector
{
  typedef std::shared_ptr<const MCxyData> DP;
  typedef std::shared_ptr<const MCfilter> FP;
  typedef std::shared_ptr<const linearModel> LMP;
  
public:

  MClinearClassifier(DP data, LMP classifier, FP filter = FP(), int const vb=MCprojector::defaultVerbose);
  ~MClinearClassifier();
  
  //changin the number of projections invalidates the predictions. 
  inline void nProj(int const np){if (m_nProj != np) {preds_ok = false;} MCprojector::nProj(np);}    
  void predict();            ///< make predictions using the model and filter. 
  PerfStruct evaluate(double threshold = 0, uint32_t min_labels = 1);         ///< evaluate predictions
  void writePreds(std::string fname = "");            ///< save predictions

  void setPruneParams(double const keep_thresh = 0.0, int const keep_size = 10 );

protected:
  
  // the model
  LMP m_model;
  
  PredictionSet m_predictions;
  bool preds_ok;

  // if there are many classes then there are many predictions, most of which we don't care about
  // to save space, only keep the top m_keep_size predictions
  // and any other predictions higher than m_keep_thresh. 
  double m_keep_thresh;
  int m_keep_size; 
  
};

struct PerfStruct
{
  double Prec1;
  double Prec5;
  double Prec10;
  double Top1;
  double Top5;
  double Top10;
  double MacroPrecision;
  double MacroRecall;
  double MacroF1;
  double MacroF1_2;
  double MicroPrecision;
  double MicroRecall;
  double MicroF1;
  double prcFeasible;
  size_t nFeasible;
};
  
#endif // MCPLINEARCLASS_H
