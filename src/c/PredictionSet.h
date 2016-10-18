#ifndef __PREDICTIONSET_H
#define __PREDICTIONSET_H

#include "typedefs.h"
#include <vector>
#include <iostream>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>

using namespace std;

typedef float predtype;

struct Prediction
{
  predtype out;
  size_t cls;
};

/** A vector<Prediction> wrapper, with \c sort, \c prune, \c predict. */
class PredVec : private std::vector<Prediction>
{
public:
    typedef std::vector<Prediction> Base;
    size_t size(){return Base::size();};
    PredVec();
    PredVec(size_t n);                          ///< init with reserve(n) memory
    ~PredVec();
    /** const base vector access */
    Base const& predvec() const {return *this;}
    void sort();                                ///< if nec, sort highest \c Prediction::out first
    /** keep highest \c Prediction::out items (at least k, with value > \c keep_thresh) */
    void prune (); //size_t k, predtype keep_thresh = boost::numeric::bounds<predtype>::highest());
    /** retrieve sorted list of \c Prediction::cls.
     * - _keep_thresh = double_max, _keep_size > 0 --> return top k classes
     * - _keep_thresh = t, _keep_size = 0  --> retrun all classes with pred > t
     * - _keep_thresh = t and _keep_size > 0  --> return all classes with pred > t, but at least k
     */
    void add_pred(predtype out, size_t cls);       ///< push back new \c Prediction item
    void add_and_prune(predtype out, size_t cls);  ///< push back new \c Prediction item keeping the vector pruned (see above)
    void set_prune_params(size_t keep_size, predtype keep_thresh=boost::numeric::bounds<predtype>::highest());
    void predict(std::vector<int>& pred_class, double thresh, size_t k=0);
    size_t getSize() const       {return Base::size();};
    void reserve(size_t n)       {Base::reserve(n);};
    void reserve_extra(size_t n) {Base::reserve(n+Base::size());};
    void free_reserved() {Base(*this).swap(*this);}
private: // utility / unused
    static bool cmp (Prediction const& a, Prediction const& b) {return a.out > b.out;};
    bool is_sorted() const {return _sorted;};
    bool is_pruned() const {return _pruned;};
private: // data
    bool _sorted;
    bool _pruned;
    size_t _keep_size;
    predtype _keep_thresh;
private:
    static bool TopWarned;
    static bool ThreshWarned;
    static bool AddWarned;
};

class PredictionSet
{
public:
    PredictionSet();
    PredictionSet(size_t n);
    template <typename EigenType> PredictionSet(const EigenType& preds);
    ~PredictionSet();

    /** generate new predicton vector for a new example.
     * maybe the entire Prediction set should e generated at once. */
    PredVec* NewPredVec(size_t n = 0);

    /** generates a new Prediction vector and puts it at position i */
    PredVec* NewPredVecAt(size_t i, size_t n=0);

    /** generates a new Prediction vector and puts it at position i.
     * does not delete previous vector at position i and does not
     * do bounds checking */
    PredVec* NewPredVecAt_fast(size_t i, size_t n = 0);

    /// return the Prediction vector of example i.
    PredVec* GetPredVec(size_t i) const;

    /// sort all predictions sets, in order of the output.
    void sort();
    size_t size() const{return _preddata->size();};

    /// prune all the Prediction vectors.
    /// keep only predictions higher than keep_thresh, but at least k predictions
    void prune();
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
private:
    std::vector<PredVec*>* _preddata;
};

// -------- inline definitions --------

inline PredVec::PredVec()
	       : Base() //std::vector<Prediction>()
	      , _sorted( true )
	      , _pruned( true )
	      , _keep_size( boost::numeric::bounds<size_t>::highest() )
	      , _keep_thresh( boost::numeric::bounds<predtype>::lowest() )
{}

inline PredVec::PredVec(size_t n) // only reserves memory, does not initialize with n predictions
	       : Base() //std::vector<Prediction>()
	      , _sorted( true )
	      , _pruned( true )
	      , _keep_size( boost::numeric::bounds<size_t>::highest() )
	      , _keep_thresh( boost::numeric::bounds<predtype>::lowest() )
{Base::reserve(n);}

inline PredVec::~PredVec()
{}

// sort predictions in order of the output
inline void PredVec::sort()
{
    if (!_sorted){
        std::sort( begin(), end(), cmp );
        _sorted=true;
    }
};

inline void PredVec::add_pred(predtype out, size_t cls)
{
  /* if (!AddWarned && (_keep_thresh > boost::numeric::bounds<predtype>::lowest() || _keep_size < boost::numeric::bounds<size_t>::highest())) */
  /*   { */
  /*     cerr << "Warning: addind a new prediction to an already pruned vector" << endl; */
  /*     AddWarned = true; */
  /*   }     */
  Base::push_back(Prediction{out,cls});
  _pruned=false;
  _sorted=false;
}

inline void PredVec::add_and_prune(predtype out, size_t cls)
{
  
  assert(~_pruned);
  
  if (Base::size() == _keep_size)
    {
      // we have exactly keepsize elements. Insert in ordered list if is in top 10.
      // Eliminate the last if under the threshold.
      predtype const last = Base::back().out;
      if (out > last)
	{
	  if (last <= _keep_thresh)
	    {
	      Base::pop_back();
	    }
	  Prediction pred = {out,cls};
	  Base::insert(std::upper_bound(Base::begin(),Base::end(),pred,cmp), pred);
	}
    }
  else if (Base::size() < _keep_size)
    {
      // we don't yet have keepsize elements. Insert in the ordered list
      Prediction pred = {out,cls};
      Base::insert(std::upper_bound(Base::begin(),Base::end(),pred,cmp), pred);
    }
  else
    {
      // we already have more than _keep_size that pass the threshold. Only add if over threshold.
      // no need for sorted list any more. 
      if(out > _keep_thresh)
	{
	  Base::push_back(Prediction{out,cls});
	  _sorted=false;
	}
    }
}

inline void PredVec::prune ()
{
  if (~_pruned)
    {
      this->sort();
      size_t new_size = std::distance( cbegin(),
				       std::lower_bound( cbegin(), cend(), Prediction{_keep_thresh,0}, cmp ));
      new_size = std::min( Base::size(), std::max( new_size, _keep_size ));
      if (new_size < Base::size()) {
        resize(new_size);
        free_reserved();
      }
    }
}

inline void PredVec::set_prune_params(size_t keep_size, predtype keep_thresh)
{
  // assert that the params have not been set to different values before;
  assert(_keep_size == boost::numeric::bounds<size_t>::highest() || _keep_size == keep_size);
  assert(_keep_thresh == boost::numeric::bounds<predtype>::lowest() || _keep_thresh == keep_thresh);
  
  _keep_size = keep_size;
  _keep_thresh = keep_thresh;
}

inline void PredVec::predict(std::vector<int>& pred_class, double thresh, size_t k)
{
    if (thresh < _keep_thresh && !ThreshWarned) {
        cerr<<"Warning: Asking for prediction threshold "<<thresh<<
            " but predictions have already been pruned using threshold "<<_keep_thresh<<endl;
        ThreshWarned = true;
    }
    if ( k > _keep_size && !TopWarned) {	
        cerr<<"Warning: Asking for top "<<k<<" but predictions have already been pruned"
            " to keep only top "<<_keep_size<<endl;
        TopWarned = true;
    }
    this->sort();
    size_t new_size = std::distance( cbegin(),
                                     std::lower_bound( cbegin(), cend(), Prediction{(predtype)thresh,0}, cmp ));
    new_size = std::min( Base::size(), std::max( new_size, k ));
    pred_class.clear();
    pred_class.reserve(new_size);
    for(size_t i=0U; i<new_size; ++i) pred_class.push_back( (*this)[i].cls );
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

// appends new Prediction vector
inline PredVec* PredictionSet::NewPredVec(size_t n)
{
  PredVec* newpredvec = new PredVec(n);
  _preddata->push_back(newpredvec);
  return newpredvec;
}

// puts a new Prediction vector at position i. Deletes old vector if exists
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


// puts a new Prediction vector at position i.
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

inline void PredictionSet::prune()
{
  for (std::vector<PredVec*>::iterator it = _preddata->begin(); it != _preddata->end(); it++)
    {
      (*it)->prune();
    }
}

#endif //__PREDICTIONSET_H
