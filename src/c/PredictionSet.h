#ifndef __PREDICTIONSET_H
#define __PREDICTIONSET_H

#include "Eigen/Sparse"
#include <vector>
#include <iostream>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include "typedefs.h"

using namespace std;

typedef float predtype;

struct prediction
{
  predtype out;
  int cls;
};

inline bool pred_comp (const prediction* a, const prediction* b) {return a->out > b->out;};

class PredVec 
{
 private:
  std::vector<prediction*>* _predvec;
  size_t _npreds;
  bool _sorted; 
  size_t _keep_size;
  predtype _keep_thresh;
  static bool TopWarned;
  static bool ThreshWarned;
  static bool AddWarned;
  void init();
 public:
  PredVec();
  PredVec(size_t n=0);
  ~PredVec();
  std::vector<prediction*>* predvec();
  void add_pred(predtype out, int cls);
  void sort();
  bool is_sorted() const {return _sorted;};
  size_t npreds() const {return _npreds;};
  size_t getSize() const {return _predvec->size();};
  void reserve_extra(size_t n) {_predvec->reserve(n+_predvec->size());};
  void reserve(size_t n) {_predvec->reserve(n);};
  void prune (size_t k, predtype keep_thresh=boost::numeric::bounds<predtype>::highest());
  void predict(std::vector<int>& pred_class, double thresh, size_t k=0);
};


class PredictionSet
{
 private:
  std::vector<PredVec*>* _preddata;
 public:
  PredictionSet();
  PredictionSet(size_t n);
  template <typename EigenType> PredictionSet(const EigenType& preds);
  ~PredictionSet();
  // generate new predicton vector for a new example.
  // maybe the entire prediction set should e generated at once. 
  PredVec* NewPredVec(size_t n = 0);
  // generates a new prediction vector and puts it at position i
  PredVec* NewPredVecAt(size_t i, size_t n=0); 
  // generates a new prediction vector and puts it at position i
  // does not delete previous vector at position i and does not
  // do bounds checking
  PredVec* NewPredVecAt_fast(size_t i, size_t n = 0);
  // return the prediction vector of example i
  PredVec* GetPredVec(size_t i) const;
  // sort all predictions sets.
  void sort();
  size_t size(){return _preddata->size();};
  // prune all the prediction vectors.
  // keep only predictions higher than keep_thresh, but at least k predictions 
  void prune(size_t k, predtype keep_thresh=boost::numeric::bounds<predtype>::highest());
  double PrecK(const SparseMb& y, int k);
  double TopK(const SparseMb& y, int k);
  void TopMetrics(double& Prec1, double& Top1,
		  double& Prec5, double& Top5, 
		  double& Prec10, double& Top10,
		  const SparseMb& y);    
  double MicroF1(const SparseMb& y, double thresh, size_t k);
  double MacroF1(const SparseMb& y, double thresh, size_t k);
  double MacroF1_2(const SparseMb& y, double thresh, size_t k);
  double MacroRecall(const SparseMb& y, double thresh, size_t k);
  void ThreshMetrics(double& MicroF1, double& MacroF1, 
		     double& MacroF1_2, 
		     double& MicroPrecision, double& MacroPrecision,
		     double& MicroRecall, double& MacroRecall, 
		     const SparseMb& y, double thresh, size_t k);    


  size_t npreds() const;
  SparseM* toSparseM() const;
};


inline void PredVec::init()
{
  _predvec = new std::vector<prediction*>;
  _npreds = 0;
  _keep_size = boost::numeric::bounds<size_t>::highest();
  _keep_thresh = boost::numeric::bounds<predtype>::lowest();
  _sorted=false;
}

inline PredVec::PredVec()
{
  init();
}

inline PredVec::PredVec(size_t n)
{
  init();
  _predvec->reserve(n);
}  

inline PredVec::~PredVec()
{
  for (std::vector<prediction*>::iterator it = _predvec->begin(); it != _predvec->end(); it++)
    {
      delete *it;
    }
  delete _predvec;
}

inline std::vector<prediction*>* PredVec::predvec()
{
  return _predvec;
}

inline void PredVec::add_pred(predtype out, int cls)
{
  /* if (!AddWarned && (_keep_thresh > boost::numeric::bounds<predtype>::lowest() || _keep_size < boost::numeric::bounds<size_t>::highest())) */
  /*   { */
  /*     cerr << "Warning: addind a new prediction to an already prunned vector" << endl; */
  /*     AddWarned = true; */
  /*   }     */
  prediction* pred = new prediction;
  pred->out = out; 
  pred->cls = cls; 
  _predvec->push_back(pred);
  _sorted=false;
  _npreds++;
}  

// sort predictions in order of the output
inline void PredVec::sort() 
{
  if (!_sorted)
    {
      std::sort(_predvec->begin(),_predvec->end(),pred_comp);
    }
  _sorted=true;
};

// keep only preds with out > keep_thresh, but at least k predictions
inline void PredVec::prune (size_t k, predtype keep_thresh)
{
  this->sort();
  //keep the largest threshold
  _keep_thresh = _keep_thresh < keep_thresh ? keep_thresh : _keep_thresh;
  //keep the smallest k
  _keep_size = _keep_size < k ? _keep_size : k;
  std::vector<prediction*>::iterator it = _predvec->begin();
  size_t new_size = 0;
  // could do this through binary search
  while (it!=_predvec->end() && (*(it++))->out > keep_thresh) new_size++;
  if (new_size < k)
    {
      new_size = k;
    }
  if (new_size < _predvec->size())
    {
      for (it = _predvec->begin() + new_size; it!=_predvec->end(); it++)
	{
	  delete *it;
	}      
      _predvec->resize(new_size);
      vector<prediction*>(*_predvec).swap(*_predvec);
    }
}

// thresh = double_max, k > 0 --> return top k classes
// thresh = t, k = 0  --> retrun all classes with pred > t
// thresh = t and k > 0  --> return all classes with pred > t, but at least k 
inline void PredVec::predict(std::vector<int>& pred_class, double thresh, size_t k)
{
  if (thresh < _keep_thresh && !ThreshWarned)
    {
      cerr << "Warning: Asking for prediction threshold " << thresh << " but predictions have already been pruned using threshold " << _keep_thresh << endl;
      ThreshWarned = true;
    }
  if ( k > _keep_size && !TopWarned)
    {	
      cerr << "Warning: Asking for top " << k << " but predictions have already been pruned to keep only top " << _keep_size << endl;
      TopWarned = true;
    }
  
  this->sort();
  
  pred_class.clear();
  std::vector<prediction*>::iterator it = _predvec->begin();
  // could do this through binary search
  size_t npreds = 0;
  while (it!=_predvec->end() && ( (*it)->out > thresh || npreds < k))
    {
      pred_class.push_back((*(it++))->cls);
      npreds++;
    }
}



inline PredictionSet::PredictionSet()
{
  _preddata = new std::vector<PredVec*>;
}

inline PredictionSet::PredictionSet(size_t n)
{
  _preddata = new std::vector<PredVec*>(n);  
}

template <typename EigenType>
inline PredictionSet::PredictionSet(const EigenType& preds)
{
  size_t n = preds.rows();
  _preddata = new std::vector<PredVec*>(n);
  for (size_t i = 0 ; i < n; i++)
    {
      PredVec* pv = NewPredVecAt_fast(i,0);
      for (typename EigenType::InnerIterator it(preds,i); it; ++it)
	{
	  pv->add_pred(it.value(),it.col());
	}
    }
}


inline PredictionSet::~PredictionSet()
{
  for (std::vector<PredVec*>::iterator it = _preddata->begin(); it != _preddata->end(); it++)
    {
      delete *it;
    }
  delete _preddata;
}

// appends new prediction vector
inline PredVec* PredictionSet::NewPredVec(size_t n)
{ 
  PredVec* newpredvec = new PredVec(n);
  _preddata->push_back(newpredvec);
  return newpredvec;
}

// puts a new prediction vector at position i. Deletes old vector if exists
inline PredVec* PredictionSet::NewPredVecAt(size_t i, size_t n)
{
  if ( _preddata->at(i) != NULL)
    {
      delete (*_preddata)[i];
    }  
  PredVec* newpredvec = new PredVec(n);
  (*_preddata)[i]=newpredvec;
  return newpredvec;
}


// puts a new prediction vector at position i.
// assumes the pointer at position i is NULL so does not delete it
// no bound checking
inline PredVec* PredictionSet::NewPredVecAt_fast(size_t i, size_t n)
{
  PredVec* newpredvec = new PredVec(n);
  (*_preddata)[i]=newpredvec;
  return newpredvec;
}

inline PredVec* PredictionSet::GetPredVec(size_t i) const
{
  return _preddata->at(i);
}


inline void PredictionSet::sort() 
{
  for (std::vector<PredVec*>::iterator it = _preddata->begin(); it != _preddata->end(); it++)
    {
      (*it)->sort();
    }
}

inline void PredictionSet::prune(size_t k, predtype keep_thresh) 
{    
  for (std::vector<PredVec*>::iterator it = _preddata->begin(); it != _preddata->end(); it++)
    {
      (*it)->prune(k, keep_thresh);
    }
}

#endif //__PREDICTIONSET_H
