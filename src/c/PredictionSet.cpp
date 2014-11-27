#include "PredictionSet.h"

bool PredVec::AddWarned = false;
bool PredVec::ThreshWarned = false;
bool PredVec::TopWarned = false;



double PredictionSet::PrecK(const SparseMb& y, int k)
{
  assert (y.rows() == _preddata->size());
  std::vector<int> preds;
  std::vector<PredVec*>::iterator predit;
  size_t i;
  double ret=0;
  for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
    {
      (*predit)->predict(preds, std::numeric_limits<predtype>::max(), k);
      // it would be more efficitent to look or the true classes in preds using find	
      size_t tp = 0;
      for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	{
	  if (y.coeff(i,*it))
	    {
	      tp++;
	    }
	}
      if (preds.size() > 0)
	{
	  ret += tp*1.0/preds.size();
	}
    }
  return ret/(_preddata->size());
}

double PredictionSet::TopK(const SparseMb& y, int k)
  {
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<PredVec*>::iterator predit;
    size_t i;
    double ret=0;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, std::numeric_limits<predtype>::max(), k);
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
    return ret/(_preddata->size());
  }
  
double PredictionSet::MicroF1(const SparseMb& y, double thresh, size_t k)
  {
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<PredVec*>::iterator predit;
    size_t i;
    size_t total_p=0;
    size_t tp = 0;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, thresh, k);
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
    double prec = tp*1.0/total_p;
    double rec = tp*1.0/y.nonZeros();
    return 2*prec*rec/(prec+rec);
  }
    
  
double PredictionSet::MacroF1(const SparseMb& y, double thresh, size_t k)
  {
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<size_t> class_tp(y.cols());
    std::vector<size_t> class_total_p(y.cols());    
    std::vector<PredVec*>::iterator predit;
    size_t i;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p[*it]++;
	    if (y.coeff(i,*it))
	      {
		class_tp[*it]++;		
	      }
	  }
      }

    // get the number of examples in each class
    std::vector<size_t> ncl(y.cols());
    for (i = 0; i < y.rows(); i++)
      {
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl[it.col()]++;
	      }
	  }
      }

    double prec = 0;
    double rec = 0;
    size_t l=0;
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
    return 2*prec*rec/(l*(prec+rec));
  }

double PredictionSet::MacroF1_2(const SparseMb& y, double thresh, size_t k)
  {
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<size_t> class_tp(y.cols());
    std::vector<size_t> class_total_p(y.cols());    
    std::vector<PredVec*>::iterator predit;
    size_t i;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p[*it]++;
	    if (y.coeff(i,*it))
	      {
		class_tp[*it]++;		
	      }
	  }
      }

    // get the number of examples in each class
    std::vector<size_t> ncl(y.cols());
    for (i = 0; i < y.rows(); i++)
      {
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl[it.col()]++;
	      }
	  }
      }

    double prec = 0;
    double rec = 0;
    double f1 = 0;
    size_t l=0;
    for (int j=0;j<y.cols();j++)
      {
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
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<size_t> class_tp(y.cols());
    std::vector<size_t> class_total_p(y.cols());    
    std::vector<PredVec*>::iterator predit;
    size_t i;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p[*it]++;
	    if (y.coeff(i,*it))
	      {
		class_tp[*it]++;		
	      }
	  }
      }

    // get the number of examples in each class
    std::vector<size_t> ncl(y.cols());
    for (i = 0; i < y.rows(); i++)
      {
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl[it.col()]++;
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
    assert (y.rows() == _preddata->size());
    std::vector<int> preds;
    std::vector<size_t> class_tp(y.cols(),0);
    std::vector<size_t> class_total_p(y.cols(),0);    
    std::vector<PredVec*>::iterator predit;
    size_t i;
    size_t tp=0;
    size_t total_p=0;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, thresh, k);
	// it woudl be more efficitent to look or the true classes in preds using find	
	for (std::vector<int>::iterator it = preds.begin(); it !=preds.end();it++)
	  {
	    class_total_p[*it]++;
	    //	    cout << *it << "   " << class_tp[*it] << "   " << class_total_p[*it] << "   " << y.coeff(i,*it) << endl;
	    if (y.coeff(i,*it))
	      {
		class_tp[*it]++;		
		tp++;
	      }
	  }
	total_p+=preds.size();
      }
    MicroPrecision = tp*1.0/total_p;
    MicroRecall = tp*1.0/y.nonZeros();
    MicroF1 = 2*MicroPrecision*MicroRecall/(MicroPrecision+MicroRecall);

    // get the number of examples in each class
    std::vector<size_t> ncl(y.cols());
    for (i = 0; i < y.rows(); i++)
      {
	for (SparseMb::InnerIterator it(y,i); it ; ++it)
	  {
	    if (it.value())
	      {
		ncl[it.col()]++;
	      }
	  }
      }

    double prec = 0;
    double rec = 0;
    double f1 = 0;
    size_t l=0;
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
    assert (y.rows() == _preddata->size());
    int maxtop = 10;
    std::vector<int> preds;
    std::vector<PredVec*>::iterator predit;
    size_t i;
    double ret=0;
    Top1 = 0; Prec1 = 0; Top5 = 0; Prec5 = 0; Top10 = 0; Prec10 = 0;
    for ( predit = _preddata->begin(), i=0; predit != _preddata->end(); predit++,i++)
      {
	(*predit)->predict(preds, std::numeric_limits<predtype>::max(), maxtop);
	if (preds.size()==0)
	  {
	    // no predictions were made for this case.
	    continue;
	  }
	size_t tp = 0;
	int k = 0;
	for (std::vector<int>::iterator it = preds.begin(); it != preds.end(); it++)
	  {	    
	    if (y.coeff(i,*it))
	      {
		tp++;
	      }
	    k++;
	    if (k == 1)
	      {
		Prec1 += tp;
		Top1 += tp;
	      }
	    if (k == 5)
	      {
		Prec5 += tp*1.0/5;
		Top5 += (tp>0);
	      }
	    if (k == 10)
	      {
		Prec10 += tp*1.0/10;
		Top10 += (tp>0);
	      }
	  }
	//k is at least 1
	if (k < 5)
	  {
	    Prec5 += tp*1.0/k;
	    Top5 += (tp>0);
	  }
	if (k < 10)
	  {
	    Prec10 += tp*1.0/k;
	    Top10 += (tp>0);
	  }	    	    
      }
    Prec1 /= _preddata->size();
    Top1 /=  _preddata->size();
    Prec5 /= _preddata->size();
    Top5 /=  _preddata->size();
    Prec10 /= _preddata->size();
    Top10 /=  _preddata->size();
  }

size_t PredictionSet::npreds() const
{
  size_t np=0;
  for (std::vector<PredVec*>::iterator it = _preddata->begin(); it != _preddata->end(); it++)
    {
      np += (*it)->getSize();
    }
}

SparseM* PredictionSet::toSparseM() const
{
  std::vector< Eigen::Triplet<double> > tripletList;
  tripletList.reserve(npreds());
  size_t i;
  size_t maxclass = 0;
  std::vector<PredVec*>::iterator predit;
  std::vector<prediction*>::iterator it;

  for(predit = _preddata->begin(),i=0; predit != _preddata->end(); predit++, i++)
    {
      for (it = (*predit)->predvec()->begin(); it != (*predit)->predvec()->end();it++)
	{
	  tripletList.push_back(Eigen::Triplet<double> (i, (*it)->cls, (*it)->out));
	  maxclass = maxclass < (*it)->cls ? (*it)->cls : maxclass;
	}
    }
  
  SparseM* m = new SparseM(_preddata->size(), maxclass+1);
  m->setFromTriplets(tripletList.begin(), tripletList.end());
  return m;
}
