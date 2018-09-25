/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "PredictionSet.h"
#include "constants.h" //MCTHREADS
#include "printing.hh"

bool PredVec::AddWarned = false;
bool PredVec::ThreshWarned = false;
bool PredVec::TopWarned = false;

double PredictionSet::PrecK(const SparseMb& y, int k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  double ret=0;
  PredictionSet* psp = this;
#if MCTHREADS
#pragma omp parallel for default(none) shared(y,psp,k) reduction(+:ret)
#endif
  for ( size_t i=0; i<psp->size(); i++)
    {	
      std::vector<int> preds;
      (*psp)[i].predict(preds, boost::numeric::bounds<predtype>::highest(), k);
      if (preds.size()==0)
	{
	  // no predictions were made for this case.
	  continue;
	}
      size_t tp = 0;
      for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	{
	  if (y.coeff(i,*it))
	    {
	      tp++;
	    }
	}
      if (preds.size()>0)
	{
	  ret += tp*1.0/preds.size();
	}
    }	
  return ret/(this->size());
}

double PredictionSet::TopK(const SparseMb& y, int k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  double ret=0;
  PredictionSet* psp = this; // omp does not take "this" 
#if MCTHREADS
#pragma omp parallel for default(none) shared(psp,y,k) reduction(+:ret)
#endif
  for ( size_t i=0; i<psp->size(); i++)
    {	
      std::vector<int> preds;
      (*psp)[i].predict(preds, boost::numeric::bounds<predtype>::highest(), k);
      // it would be more efficitent to look or the true classes in preds using find	
      for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	{
	  if (y.coeff(i,*it))
	    {
	      ret++;
	      break;
	    }
	}
    }
  return ret/(this->size());
}

double PredictionSet::MicroF1(const SparseMb& y, double thresh, size_t k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  size_t total_p=0;
  size_t tp = 0;
  PredictionSet* psp = this; //needed to make omp work
#if MCTHREADS
#pragma omp parallel default(none) shared(psp,thresh,k,y) reduction(+:total_p,tp)
#endif
  {
#if MCTHREADS
#pragma omp for
#endif
    for ( size_t i=0; i < psp->size(); i++)
      {
	std::vector<int> preds;
	(*psp)[i].predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    if (y.coeff(i,*it))
	      {
		tp++;
	      }
	  }
	total_p += preds.size();
      }
  }
  double prec = tp*1.0/total_p;
  double rec = tp*1.0/y.nonZeros();
  return 2*prec*rec/(prec+rec);
}
  
double PredictionSet::MacroF1(const SparseMb& y, double thresh, size_t k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  std::vector<size_t> class_tp(y.cols());
  std::vector<size_t> class_total_p(y.cols());    
  std::vector<size_t> ncl(y.cols());
  PredictionSet* psp = this; //needed to make omp work
#if MCTHREADS
#pragma omp parallel default(none) shared(psp,thresh,k,y,class_tp,class_total_p,ncl)
#endif
  {
    std::vector<size_t> class_total_p_private(y.cols());    
    std::vector<size_t> class_tp_private(y.cols(),0);
    std::vector<size_t> ncl_private(y.cols(),0);
#if MCTHREADS
#pragma omp for
#endif
    for ( size_t i=0; i < psp->size(); i++)
      {
	std::vector<int> preds;
	(*psp)[i].predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p_private[*it]++;
	    if (y.coeff(i,*it))
	      {
		class_tp_private[*it]++;		
	      }
	  }      
    
	// get the number of examples in each class
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl_private[it.col()]++;
	      }
	  }
      }
#if MCTHREADS
#pragma omp critical
#endif
    {
      for (size_t i=0; i < y.cols(); i++)
	{
	  class_total_p[i] += class_total_p_private[i];
	  class_tp[i] += class_tp_private[i];
	  ncl[i]+=ncl_private[i];
	}
    }
  }

  double prec = 0;
  double rec = 0;
  size_t l=0;
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:prec,rec,l)
#endif
  for (int j=0;j<y.cols();j++)
    {
      if (class_total_p[j] > 0)
	{
	  prec += class_tp[j]*1.0/class_total_p[j];
	}
      if (ncl[j] > 0)
	{
	  rec += class_tp[j]*1.0/ncl[j];
	  l++;
	}
    }
  return prec+rec?2*prec*rec/(l*(prec+rec)):0;
}

double PredictionSet::MacroF1_2(const SparseMb& y, double thresh, size_t k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  std::vector<size_t> class_tp(y.cols());
  std::vector<size_t> class_total_p(y.cols());    
  std::vector<size_t> ncl(y.cols());
  PredictionSet* psp = this; //needed to make omp work
#if MCTHREADS
#pragma omp parallel default(none) shared(psp,thresh,k,y,class_tp,class_total_p,ncl)
#endif
  {
    std::vector<size_t> class_total_p_private(y.cols());    
    std::vector<size_t> class_tp_private(y.cols(),0);
    std::vector<size_t> ncl_private(y.cols(),0);
#if MCTHREADS
#pragma omp for
#endif
    for ( size_t i=0; i < psp->size(); i++)
      {
	std::vector<int> preds;
	(*psp)[i].predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p_private[*it]++;
	    if (y.coeff(i,*it))
	    {
	      class_tp_private[*it]++;		
	    }
	  }
	
	// get the number of examples in each class
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl_private[it.col()]++;
	      }
	  }
      }
#if MCTHREADS
#pragma omp critical
#endif
    {
      for (size_t i=0; i < y.cols(); i++)
	{
	  class_total_p[i] += class_total_p_private[i];
	  class_tp[i] += class_tp_private[i];
	  ncl[i]+=ncl_private[i];
	}
    }
  }
  
  double f1 = 0;
  size_t l=0;
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:f1,l)
#endif
  for (int j=0;j<y.cols();j++)
    {
      double prec = 0;
      double rec = 0;      
      if (class_total_p[j] > 0)
	{
	  prec = class_tp[j]*1.0/class_total_p[j];
	}
      if (ncl[j] > 0)
	{
	  rec = class_tp[j]*1.0/ncl[j];
	  if (prec+rec>0)
	    {
	      f1 += 2*prec*rec/(prec+rec);
	    }
	  l++;
	}
    }
  return f1/l;
}

  
double PredictionSet::MacroRecall(const SparseMb& y, double thresh, size_t k)
  {
    assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
    std::vector<size_t> class_tp(y.cols());
    std::vector<size_t> ncl(y.cols());
    PredictionSet* psp = this; //needed to make omp work
#if MCTHREADS
#pragma omp parallel default(none) shared(psp,thresh,k,y,class_tp,ncl)
#endif
    {
      std::vector<size_t> class_tp_private(y.cols(),0);
      std::vector<size_t> ncl_private(y.cols(),0);
#if MCTHREADS
#pragma omp for
#endif
      for ( size_t i=0; i < psp->size(); i++)
	{
	  std::vector<int> preds;
	  (*psp)[i].predict(preds, thresh, k);
	  // it woudl be more efficitent to look or the true classes in preds using find	
	  for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	    {
	      if (y.coeff(i,*it))
		{
		  class_tp_private[*it]++;		
		}
	    }
	}
      // get the number of examples in each class
      for (size_t i = 0; i < y.rows(); i++)
	{
	  for (SparseMb::InnerIterator it(y,i); it ; ++it)
	    {
	      if (it.value())
		{
		  ncl_private[it.col()]++;
		}
	    }
	}
      
#if MCTHREADS
#pragma omp critical
#endif
      {
	for (size_t i=0; i < y.cols(); i++)
	  {
	    class_tp[i] += class_tp_private[i];
	    ncl[i]+=ncl_private[i];
	  }
      }
    }
    double rec = 0;
    size_t l=0;
    for (int j=0;j<y.cols();j++)
      {
	if (ncl[j] > 0)
	  {	    
	    rec += class_tp[j]*1.0/ncl[j];
	    l++;
	  }
      }
    return rec/l;
  }

void PredictionSet::ThreshMetrics(double& MicroF1, double& MacroF1, 
			       double& MacroF1_2, 
			       double& MicroPrecision, double& MacroPrecision,
			       double& MicroRecall, double& MacroRecall, 
			       const SparseMb& y, double thresh, size_t k)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  std::vector<size_t> class_tp(y.cols(),0);
  std::vector<size_t> class_total_p(y.cols(),0);    
  std::vector<size_t> ncl(y.cols(),0);
  size_t tp=0;
  size_t total_p=0;
  PredictionSet* psp = this; //needed to make omp work  
#if MCTHREADS
#pragma omp parallel default(none) shared(psp,thresh,k,y,class_tp,class_total_p,total_p,tp,ncl)
#endif
  {
    std::vector<size_t> class_tp_private(y.cols(),0);
    std::vector<size_t> class_total_p_private(y.cols(),0);    
    std::vector<size_t> ncl_private(y.cols(),0);
    size_t tp_private=0;
    size_t total_p_private=0;
#if MCTHREADS
#pragma omp for
#endif
    for ( size_t i=0; i < psp->size(); i++)
      {
	std::vector<int> preds;
	(*psp)[i].predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p_private[*it]++;
	    if (y.coeff(i,*it))
	      {
		class_tp_private[*it]++;		
		tp_private++;
	      }
	  }
	
	// get the number of examples in each class
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl_private[it.col()]++;
	      }
	  }	  
	total_p_private+=preds.size();
      }
#if MCTHREADS
#pragma omp critical
#endif
    {
      for (size_t i=0; i < y.cols(); i++)
	{
	  class_total_p[i] += class_total_p_private[i];
	  class_tp[i] += class_tp_private[i];
	  ncl[i]+=ncl_private[i];
	}
      tp+=tp_private;
      total_p += total_p_private;
    }
  }
  
  MicroPrecision = tp*1.0/total_p;
  MicroRecall = tp*1.0/y.nonZeros();
  MicroF1 = 2*MicroPrecision*MicroRecall/(MicroPrecision+MicroRecall);
  
  double prec = 0;
  double rec = 0;
  double f1 = 0;
  size_t l=0;
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:prec,rec,f1,l)
#endif
  for (int j=0;j<y.cols();j++)
    {
      double p=0,r=0;
      if (class_total_p[j] > 0)
	{	    
	  p = class_tp[j]*1.0/class_total_p[j];
	  prec += p;
	}
      if (ncl[j] > 0)
	{	    
	  r = class_tp[j]*1.0/ncl[j];
	  rec += r;
	  if (p+r>0)
	    {
	      f1 += 2*p*r/(p+r);
	    }
	  l++;
	}
    }
  MacroPrecision = prec*1.0/l;
  MacroRecall = rec*1.0/l;
  MacroF1 = 2*MacroPrecision*MacroRecall/(MacroPrecision+MacroRecall);
  MacroF1_2 = f1/l;
}

void PredictionSet::TopMetrics(double& Prec1, double& Top1,
			       double& Prec5, double& Top5, 
			       double& Prec10, double& Top10,
			       const SparseMb& y)
{
  assert (y.rows() == static_cast<SparseMb::Index>(this->size()));
  int maxtop = 10;
  double my_top1=0, my_prec1=0, my_top5=0, my_prec5=0, my_top10=0, my_prec10=0;
  PredictionSet* psp = this; //needed to make omp work
#if MCTHREADS
#pragma omp parallel for default(none) shared(psp,y,maxtop) reduction(+:my_top1,my_top5,my_top10,my_prec1,my_prec5,my_prec10)
#endif
  for ( size_t i=0; i<psp->size(); i++)
    {	
      std::vector<int> preds;
      (*psp)[i].predict(preds, boost::numeric::bounds<predtype>::highest(), maxtop);
      if (preds.size()==0)
	{
	  // no predictions were made for this case.
	  continue;
	}
      
      size_t tp = 0;
      int k = 0;
      for (std::vector<int>::iterator it = preds.begin(); it != preds.end() && k<=10; it++)
	{	    
	  if (y.coeff(i,*it))
	    {
	      tp++;
	    }
	  k++;
	  if (k == 1)
	    {
	      my_prec1 += tp;
	      my_top1 += tp;
	    }
	  if (k == 5)
	    {
	      my_prec5 += tp*0.2;
	      my_top5 += (tp>0);
	    }
	  if (k == 10)
	    {
	      my_prec10 += tp*0.1;
	      my_top10 += (tp>0);
	    }
	}
      //k is at least 1 (tested for empty preds vector above)
      if (k < 5)
	{
	  my_prec5 += tp*1.0/k;
	  my_top5 += (tp>0);
	}
      if (k < 10)
	{
	  my_prec10 += tp*1.0/k;
	  my_top10 += (tp>0);
	}	    	    
    }      
  
  Prec1 = my_prec1 / this->size();
  Top1 = my_top1 / this->size();
  Prec5 = my_prec5 / this->size();
  Top5 =  my_top5 / this->size();
  Prec10 = my_prec10 / this->size();
  Top10 = my_top10 / this->size();
}

size_t PredictionSet::npreds() const
{
  size_t np=0;
  for (PV::const_iterator it = PV::begin(); it != PV::end(); it++)
    {
      np += (*it).getSize();
    }
  return np;
}

size_t PredictionSet::maxclass() const
{
  size_t mc = 0;
  PV::const_iterator predit;
  
  // find the number of classes (or at least the highest class with a prediction)
  for(predit = PV::begin(); predit != PV::end(); predit++)
    {
      for( auto const& p: (*predit).predvec() ){
	mc = std::max( mc, p.cls );
      }
    }  
  return mc;
}

SparseM* PredictionSet::toSparseM() const
{
  std::vector< Eigen::Triplet<double> > tripletList;
  tripletList.reserve(npreds());
  size_t i;
  size_t maxclass = 0;
  PV::const_iterator predit;

  for(predit = PV::begin(),i=0; predit != PV::end(); predit++, i++)
    {
      for( auto const& p: (*predit).predvec() ){
	tripletList.push_back( Eigen::Triplet<double>(i, p.cls, p.out));
	maxclass = std::max( maxclass, p.cls );
      }
    }
  
  SparseM* m = new SparseM(this->size(), maxclass+1);
  m->setFromTriplets(tripletList.begin(), tripletList.end());
  return m;
}

DenseM* PredictionSet::toDenseM() const
{
  size_t i;	
  DenseM* m = new DenseM(this->size(), maxclass()+1);
  PV::const_iterator predit;

  for(predit = PV::begin(),i=0; predit != PV::end(); predit++, i++)
    {
      for( auto const& p: (*predit).predvec() ){
	m->coeffRef(i,p.cls)=p.out;
      }
    }
  return m;

}

void PredictionSet::write(std::ostream& out, bool const dense /*=false*/) const
{
  if (dense)
    {
      DenseM* dm = toDenseM();
      detail::io_txt(out,*dm);
      delete dm;
    }
  else
    {
      SparseM* sm = toSparseM();
      detail::io_txt(out,*sm);
      delete sm;
    }
}
