/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __MCXYDATA_H
#define __MCXYDATA_H

#include "typedefs.h"
#include <array>

/** NEW: introduce a data wrapper.
 *
 * - This allows base x [,y] data to be passed/shared easily between solver
 *   and projector objects.
 * - Another benefit is removing duplicate code for reading data.
 * - Eventually, may help move template code into the library? */
class MCxyData {
public:
    MCxyData(int const verb = 0);
    MCxyData(DenseM const& x, int const verb = 0);   
    MCxyData(DenseM const& x, SparseMb const& y, int const verb = 0);
    MCxyData(SparseM const& x, int const verb = 0);   
    MCxyData(SparseM const& x, SparseMb const& y, int const verb = 0);
    
    ~MCxyData(){};
    /// \name row-wise test data matrix
    //@{
    // perhaps denseOk and sparseOk can be replaced by xDense.size() != 0 (etc.) ?
    DenseM xDense;
    bool denseOk;
    SparseM xSparse;
    bool sparseOk;
    //@}
    SparseMb y;                 ///< optional for projection operation.
    /// \name optional, private stats
    //@{
private:
    double qscal; ///< if >0, the multiplier used for \c quadx dimensions
    double xscal; ///< if >0, the global x multipler used for \c xscale
    int verbose;


public:
    //@}
    void toDense();
    void toSparse();
    
    void read (std::string xFile, std::string yFile="");  ///< read x and y
    void xread( std::string xFile );    ///< read x (binary, sparse/dense, libsvm, xml) 
    void yread( std::string yFile );    ///< read y (sparse binary or text)
    void xwrite( std::string xFile ) const; ///< save x (binary only, for now)
    void ywrite( std::string yFile, bool bin = true ) const; ///< save y

    std::string shortMsg() const;       ///< format+dimensions

    void xunitnormal();   // make x rows into unit vectors by scaling

    /*  remove mean(if center=true),stdev from x cols (colNorm=true) or rows (colNorm=false)(dense only)  */	
    void xstdnormal(bool colNorm = true, bool center = true);  

    /*  remove mean(if center=true),stdev from x cols (colNorm=true) or rows (colNorm=false)(dense only)
    **  return the removed mean and stdev in the mean and stdev vectors. 
    ** if useMeanStdev = true use mean and stdev for normalization rather than the true mean and stdev*/	
    void xstdnormal(Eigen::VectorXd& mean, Eigen::VectorXd& stdev, bool colNorm = true, bool center = true, bool useMeanStdev = false); 

    void xscale(double scal);           ///< multiply all x values by const
    double xmul() const {return xscal;} ///< what's global x multiplier?
 
    //remove features that appear in fewer than minex examples 
    void removeRareFeatures(const int minex=1);
    // remove features that appear in fewer than minex examples. Return a feature map between the new and all columns.
    // if useFeatureMap is true, the provided feature map (and reverse feature map) are used to remove features (minex
    //   is ignored)
    void removeRareFeatures(std::vector<std::size_t>& feature_map, std::vector<std::size_t>& reverse_feature_map, const int minex=1, const bool useFeatureMap=false );
    //remove labels that appear in fewer than minex examples 
    void removeRareLabels(const int minex=1);
    // remove labels that appear in fewer than minex examples. Return a label map between the new and all labels.
    // if useLabelMap is true, the provided label map (and reverse label map) are used to remove labels (minex
    //   is ignored)
    // does not remove exmaples that are left with zero labels. 
    void removeRareLabels(std::vector<std::size_t>& label_map, std::vector<std::size_t>& reverse_label_map, const int minex=1, const bool useLabelMap=false );
   
    void quadx(double qscal=0.0);       ///< add quadratic dimensions (0.0 autoscales, somehow) \throw if no x data
    double quadmul() const {return qscal;} ///< return the used quadmul (or 0.0 if quadx has not been called)
    
    // I got annoyed with weighting for an error before aborting trying binary
    // reads. So let me (everywhere, sigh) use a magic header, for a quick check.
    static std::array<char,4> magic_xSparse; ///< 0x00,'X','s','4'  // values saved as floats
    static std::array<char,4> magic_xDense;  ///< 0x00,'X','d','4'  // values saved as floats
    static std::array<char,4> magic_yBin;    ///< 0x00,'Y','s','b'
    // feel free to add any other [binary] formats.
};

inline void MCxyData::toDense()
{
  if (!denseOk)
    {
      if(sparseOk)
	{
	  xDense = xSparse;
	  denseOk=true;
	  xSparse = SparseM();
	  sparseOk = false;
	}
      else
	{
	  throw std::runtime_error("toDense called without valid data");
	}
    }
}

inline void MCxyData::toSparse()
{
  if (!sparseOk)
    {
      if(denseOk)
	{
	  xSparse = xDense.sparseView();
	  sparseOk=true;
	  xDense = DenseM();
	  denseOk = false;
	}
      else
	{
	  throw std::runtime_error("toSparse called without valid data");
	}
    }
}

#endif //__MCXYDATA_H
