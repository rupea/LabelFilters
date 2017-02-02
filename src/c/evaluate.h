#ifndef _EVALUATE_H
#define _EVALUATE_H

#include "typedefs.h"
#include "predict.h"

typedef std::vector<PredictionSet*> predvec;
typedef std::vector<double> doublevec;

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

template <typename EigenType, typename ovaType>
void predict_chunk(predvec& predictions, VectorXsz& no_active,
		   doublevec& filter_time, doublevec& predict_time,
		   doublevec& total_time,
		   const EigenType& x, 
		   const DenseColM* wmat, const DenseColM* lmat,
		   const DenseColM* umat,		   
		   const ovaType& ovaW_chunk, 
		   const size_t start_class, predtype thresh, int k,
		   bool allproj, bool verbose);

template <typename EigenType>
class predict_chunk_visitor : public boost::static_visitor<>
{
 public:
  //  template<typename EigenType>
 predict_chunk_visitor(predvec& predictions, VectorXsz& no_active,
		       doublevec& filter_time, doublevec& predict_time,
		       doublevec& total_time,
		       const EigenType& x, 
		       const DenseColM* wmat, const DenseColM* lmat,
		       const DenseColM* umat,		   
		       const size_t start_class, predtype thresh, int k,
		       bool allproj, bool verbose) :
  predictions(predictions), no_active(no_active), filter_time(filter_time), predict_time(predict_time), total_time(total_time), x(x), wmat(wmat), lmat(lmat), umat(umat), start_class(start_class), thresh(thresh), k(k), allproj(allproj), verbose(verbose) {};
  template <typename ovaType>
    void operator() (const ovaType& ovaW_chunk) const
    {
      predict_chunk(predictions, no_active, filter_time, predict_time, total_time,
		    x, wmat, lmat, umat, ovaW_chunk, 
		    start_class, thresh, k, allproj, verbose);
    };
  
 private:
  predvec& predictions;
  VectorXsz& no_active;
  doublevec& filter_time;
  doublevec& predict_time;
  doublevec& total_time;
  const EigenType& x; 
  const DenseColM* wmat; 
  const DenseColM* lmat;
  const DenseColM* umat;		   
  const size_t start_class;
  predtype thresh;
  int k;
  bool allproj;
  bool verbose;
};


template <typename EigenType> inline
void evaluate_projection(const std::vector<EigenType*>& x, 
			 const std::vector<SparseMb*>& y, 
			 const ovaModel& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 const std::vector<std::string>& setnames,
			 bool allproj, bool verbose, ostream& out = cout);


template <typename EigenType>
void evaluate_projection_chunks(const std::vector<EigenType*>& x, 
				const std::vector<SparseMb*>& y, 
				const std::string& ovaFile, const std::string& ovaFormat, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				const std::vector<std::string>& setnames,
				bool allproj, bool verbose, ostream& out = cout);



template <typename EigenType>
void get_projection_measures(const EigenType& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue);


// -------- inline template definitions --------> evaluate.hh (not needed if linking with library)

#endif //_EVALUATE_H
