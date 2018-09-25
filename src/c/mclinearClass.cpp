/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mclinearClass.h"
#include "mcxydata.h"
#include "printing.hh"
#include "utils.h"              // OUTWIDE
#include "linearModel.h"
#include "linearModel.hh"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory> //shared_ptr

using namespace std;

MClinearClassifier::MClinearClassifier(DP data, LMP classifier, FP filter /*= FP()*/, int const vb /*=0*/)
  : MCprojector(data, filter, vb)
  , m_model(classifier)
  , m_predictions()
  , preds_ok(false)
  , m_keep_thresh(0.0)
  , m_keep_size(10)      
{
}

MClinearClassifier::~MClinearClassifier(){}

void MClinearClassifier::predict()
{
  if(verbose>=1) cout<<" MClinearClassifier:predict() "<<(m_xy->denseOk?"dense":m_xy->sparseOk?"sparse":"HUH?")<<endl;
  
  if (preds_ok) 
    {
      if (verbose >= 1)
	{      
	  cout << "   Preditions have already been calculated. Returning.";
	}
      return;
    }
    
  if (!m_xy)
    {	
      throw runtime_error("can not predict witout data");
    }

  if (!m_model)
    {
      throw runtime_error("can not predict witout model");
    }
    
  if (!feasible_ok)
    {
      runFilter();      
    }  
  if( m_xy->denseOk ){
    // number of feasible labels is calculated almost for free here, so do it
    m_nfeasible = m_model->predict(m_predictions, m_xy->xDense, m_feasible, verbose, m_keep_thresh, m_keep_size);
  }else if( m_xy->sparseOk ){
    m_nfeasible = m_model->predict(m_predictions, m_xy->xSparse, m_feasible, verbose, m_keep_thresh, m_keep_size);
  }else{
    throw std::runtime_error("neither sparse nor dense training x was available");
  }
  preds_ok = true;
}


void MClinearClassifier::setPruneParams(double const keep_thresh /*= 0.0*/, int const keep_size /*= 10*/ )
{
  m_keep_size = keep_size;
  m_keep_thresh = keep_thresh;
  if (preds_ok)
    { 
      if (keep_thresh >= m_keep_thresh && keep_size <= m_keep_size)
	{
	  //already have the predictions. We only need to further prune them
	  m_predictions.set_prune_params(keep_size, keep_thresh);
	  m_predictions.prune();
	}
      else
	{
	  //we are asking for more predictions than we have. Need to redo the predictions. 
	  preds_ok = false;
	  predict();
	}
    }
}
    
PerfStruct MClinearClassifier::evaluate(double threshold/*=0.0*/, uint32_t min_labels/*=1*/)
{

  PerfStruct perfs;
  if (!preds_ok)
    {
      predict();
    }
    
  m_predictions.TopMetrics(perfs.Prec1, perfs.Top1, perfs.Prec5, perfs.Top5, perfs.Prec10, perfs.Top10, m_xy->y);
  m_predictions.ThreshMetrics(perfs.MicroF1, perfs.MacroF1, perfs.MacroF1_2, perfs.MicroPrecision, perfs.MacroPrecision, perfs.MicroRecall, perfs.MacroRecall, m_xy->y, threshold, min_labels);
    
  nFeasible(perfs.nFeasible, perfs.prcFeasible);
    
  return perfs;
}

void MClinearClassifier::writePreds(std::string fname /*=""*/)
{    
  if(verbose>=1) cout<<" MClinearClassifier::writePreds()"
		     <<"\tSaving predictions" <<endl;
  if (!preds_ok)
    {
      throw runtime_error("writePreds was called, but predictions have not been made yet");
    }
  
  ofstream ofs;
  if (fname!="")
    {
      ofs.open(fname);
    }
  
  ostream& out = fname==""?cout:ofs;
 
  m_predictions.write(out);    
}

  
