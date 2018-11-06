/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __PREDICTIONSET_H
#define __PREDICTIONSET_H

#include "typedefs.h"
#include <vector>
#include <iostream>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>

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
    void set_prune_params(size_t keep_size=boost::numeric::bounds<predtype>::highest(), predtype keep_thresh=boost::numeric::bounds<predtype>::lowest());
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
    bool _preds_removed;
    size_t _keep_size;
    predtype _keep_thresh;    
private:
    static bool TopWarned;
    static bool ThreshWarned;
    static bool AddWarned;
};

class PredictionSet:public std::vector<PredVec>
{
  typedef std::vector<PredVec> PV;

public:
    PredictionSet();
    PredictionSet(size_t n);
    template <typename EigenType> PredictionSet(const EigenType& preds);
    ~PredictionSet();

    /// sort all predictions sets, in order of the output.
    void sort();

    /// prune all the Prediction vectors.
    /// keep only predictions higher than keep_thresh, but at least k predictions
    void prune();
    void set_prune_params(size_t keep_size=boost::numeric::bounds<predtype>::highest(), predtype keep_thresh=boost::numeric::bounds<predtype>::lowest());
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

    SparseM* toSparseM() const;
    DenseM* toDenseM() const;
    void write(std::ostream& out, bool const dense=false) const;
    
 private:
    size_t npreds() const;
    size_t maxclass() const;
};

// -------- inline definitions --------

inline PredVec::PredVec()
	       : Base() //std::vector<Prediction>()
	      , _sorted( true )
	      , _pruned( true )
	      , _preds_removed( false )
	      , _keep_size( boost::numeric::bounds<size_t>::highest() )
	      , _keep_thresh( boost::numeric::bounds<predtype>::lowest() )
{}

inline PredVec::PredVec(size_t n) // only reserves memory, does not initialize with n predictions
	       : Base() //std::vector<Prediction>()
	      , _sorted( true )
	      , _pruned( true )
	      , _preds_removed( false )
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
  if (!AddWarned && _preds_removed)
    {
      std::cerr << "Warning: addind a new prediction to an already pruned vector" << std::endl;
      AddWarned = true; 
    }    
  Base::push_back(Prediction{out,cls});
  _pruned=false;
  _sorted=false;
}

inline void PredVec::add_and_prune(predtype out, size_t cls)
{
  if (Base::size() == _keep_size)
    {
      // we have exactly keepsize elements. Insert in ordered list if is in top 10.
      // Eliminate the last if under the threshold.
      sort();
      predtype const last = Base::back().out;
      if (out > last)
	{
	  if (last <= _keep_thresh)
	    {
	      Base::pop_back();
	      _preds_removed = true;
	    }
	  // insert the new prediction in the right place to keep the list sorted.
	  Prediction pred = {out,cls};
	  Base::insert(std::upper_bound(Base::begin(),Base::end(),pred,cmp), pred);	 
	}
    }
  else if (Base::size() < _keep_size)
    {
      // we don't yet have keepsize elements. Add the new prediction.
      Prediction pred = {out,cls};
      Base::push_back(pred);
      _sorted=false;
    }
  else
    {
      // we already have more than _keep_size that pass the threshold. Only add if over threshold.
      // no need for sorted list any more. 
      if(out > _keep_thresh)
	{
	  prune();
	  Base::push_back(Prediction{out,cls});
	  _sorted=false;
	}
      else
	{
	  _preds_removed = true;
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
	_preds_removed = true;
        resize(new_size);
        free_reserved();
      }
    }
  _pruned = true;
  
}

inline void PredVec::set_prune_params(size_t keep_size, predtype keep_thresh)
{

  if ( !TopWarned && keep_size > _keep_size && _preds_removed )
    {
      std::cerr<<"Warning: Asking for top "<<keep_size<<" but predictions have already been pruned"
	" to keep only top "<<_keep_size<<std::endl;
      TopWarned = true;
    }
  if ( !ThreshWarned && keep_thresh < _keep_thresh && _preds_removed )      
    {
      std::cerr<<"Warning: Asking for prediction threshold "<<keep_thresh<<
	" but predictions under " << _keep_thresh << " have already been pruned."<<std::endl;
      ThreshWarned = true;
    }
  _keep_size = keep_size;
  _keep_thresh = keep_thresh;
}

inline void PredVec::predict(std::vector<int>& pred_class, double thresh, size_t k)
{
    if (thresh < _keep_thresh && !ThreshWarned) {
        std::cerr<<"Warning: Asking for prediction threshold "<<thresh<<
            " but predictions have already been pruned using threshold "<<_keep_thresh<<std::endl;
        ThreshWarned = true;
    }
    if ( k > _keep_size && !TopWarned) {	
        std::cerr<<"Warning: Asking for top "<<k<<" but predictions have already been pruned"
            " to keep only top "<<_keep_size<<std::endl;
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



inline PredictionSet::PredictionSet():PV()
{
}

inline PredictionSet::PredictionSet(size_t n): PV(n)
{
}

template <typename EigenType>
inline PredictionSet::PredictionSet(const EigenType& preds):PV(preds.rows())
{
  for (size_t i = 0 ; i < preds.rows(); i++)
    {
      for (typename EigenType::InnerIterator it(preds,i); it; ++it)
	{
	  (*this)[i].add_pred(it.value(),it.col());
	}
    }
}


inline PredictionSet::~PredictionSet()
{
}



inline void PredictionSet::sort()
{
  // add paralelization?
  for (PV::iterator it = PV::begin(); it != PV::end(); it++)
    {
      (*it).sort();
    }
}

inline void PredictionSet::set_prune_params(size_t keep_size/*=boost::numeric::bounds<predtype>::highest()*/,
					    predtype keep_thresh/*=boost::numeric::bounds<predtype>::lowest()*/)
{
  for (PV::iterator it = PV::begin(); it != PV::end(); it++)
    {
      (*it).set_prune_params(keep_size,keep_thresh);
    }
}

inline void PredictionSet::prune()
{
  for (PV::iterator it = PV::begin(); it != PV::end(); it++)
    {
      (*it).prune();
    }
}

#endif //__PREDICTIONSET_H
