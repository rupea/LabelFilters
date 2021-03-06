/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __LINEARMODEL_H
#define __LINEARMODEL_H

#include "linearModel_detail.h"
#include "typedefs.h"
#include "PredictionSet.h"
#include <boost/limits.hpp>
#include <array>

/** A linear model. 
 *  - The model is stored as a column-major float matrix of ndim x nclass
 *  - Accomodates both dense and sparse models. 
 **/


class linearModel{
public:
    linearModel();
    linearModel(ovaDenseColM const& W, Eigen::RowVectorXd const& intercept = Eigen::RowVectorXd());   
    linearModel(ovaSparseColM const& W, Eigen::RowVectorXd const& intercept = Eigen::RowVectorXd());   
    
    ~linearModel(){};
    /// \name col-wise model matrix W
    //@{
    // perhaps denseOk and sparseOk can be replaced by WDense.size() != 0 (etc.) ?
    ovaDenseColM WDense;
    bool denseOk;
    ovaSparseColM WSparse;
    bool sparseOk;
    Eigen::RowVectorXd intercept;  //make it vector of doubles to avoid type casting in linearmodel_detail::predict 
    //@}
    std::istream& read_sparse_txt(std::istream& is);
    
public:
    void toDense();
    void toSparse();

    void read( std::string modelFile );    ///< read (binary sparse/dense; text sparse ) using magic header for binary
    void read( std::ifstream& ifs, bool sparse, bool binary); ///< read without magic header; Keep exposed in case binary files without magic header need to be read
    void write( std::string modelFile, bool binary = true ) const; ///< save

    template <typename Eigentype>
      size_t predict (PredictionSet& predictions, Eigentype const& x, ActiveSet const* feasible = NULL, bool verbose = false, predtype keep_thresh = boost::numeric::bounds<predtype>::lowest(), size_t keep_size = boost::numeric::bounds<size_t>::highest()) const; // predict x*w  
    
    // I got annoyed with weighting for an error before aborting trying binary
    // reads. So let me (everywhere, sigh) use a magic header, for a quick check.
    static std::array<char,4> magic_Sparse; ///< 0x00,'W','s','4'  //values saved as floats
    static std::array<char,4> magic_Dense;  ///< 0x00,'W','d','4'  //values saved as floats
    // feel free to add any other [binary] formats.
};

inline void linearModel::toDense()
{
  if (!denseOk)
    {
      if(sparseOk)
	{
	  WDense = WSparse;
	  denseOk=true;
	  WSparse = ovaSparseColM();
	  sparseOk = false;
	}
      else
	{
	  throw std::runtime_error("toDense called without valid model");
	}
    }
}

inline void linearModel::toSparse()
{
  if (!sparseOk)
    {
      if(denseOk)
	{
	  WSparse = WDense.sparseView();
	  sparseOk=true;
	  WDense = ovaDenseColM();
	  denseOk = false;
	}
      else
	{
	  throw std::runtime_error("toSparse called without valid model");
	}
    }
}

#endif //__LINEAEMODEL_H


