/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCPROJECTOR_H
#define MCPROJECTOR_H
/** \file
 * Class that applies a label filter to a dataset, and manages the active labels set 
 */

#include "typedefs.h"
#include <memory>

class MCxyData;
class MCfilter;

class MCprojector
{
  // If the data or the filter changes, the feasible set is invalidated
  // hence the const to make sure they are not changed outside the class
  typedef std::shared_ptr<const MCxyData> DP;
  typedef std::shared_ptr<const MCfilter> FP;
public:
  static const int defaultVerbose=1;
  
  MCprojector( DP data, FP filter = FP(), int const vb=defaultVerbose );    
  ~MCprojector();
  
  void runFilter();            ///< project data and fiter classes
  void saveFeasible( std::string const& fname = "", bool const binary = false);  ///< output fesible set to fname (or cout if fname == "")
  
  inline FP const& filter() const {return m_filter;}
  inline DP const& data() const {return m_xy;}
  inline void nProj(int const np){if (m_nProj != np) {m_nProj = np; feasible_ok=false; m_nfeasible = 0;}}    
  inline int nProj(){ return m_nProj;}

  //calculate the total number of feasible labels for the dataset
  void nFeasible(size_t& nfeasible, double& prc_feasible); 

protected:
  
  // data (y is optional for filtering, and is used only in evaluation). 
  DP m_xy;
  
  // Label filter
  FP m_filter;
  
  /** \b Set of feasible labels for each instance. 
   * For each example row of \c x{Dense|Sparse},
   * raw output is a bitset of feasible classes.
   * A null value with feasible_ok = true means that nothing is filtered. 
   */
  
  ActiveSet* m_feasible; //may be null if no filter is used.
  bool feasible_ok;

  int m_nProj;       ///< number of projetions to use for the label filter  [-1=all projections, 0 = no fltering]
  size_t m_nfeasible; ///< total nubmer of feasible labels. Calculated at prediction time or by nFeasible fucntion. 0 means it has not been calculated yet.
  int verbose;            ///< verbosity
};
#endif // MCPROJECTOR_H
