#ifndef MCLEARNFILTER_H
#define MCLEARNFILTER_H
/** \file
 * Encapsulate original standalone mcsolve/mcproj into library.
 */
#include "mcsolver.h"
#include "parameter-args.h"
#include <memory>

class MCxyData; 

/** high level MCsolver api, that encapsulates the  parameters of the solver and the data used to leran the filter */
class MClearnFilter: public MCsolver {
  typedef std::shared_ptr<const MCxyData> DP;
  typedef MCsolver S;
  
 public:
  MClearnFilter(DP data, MCsoln const& sol = MCsoln(), SolverParams const& parms = SolverParams());
  
  ~MClearnFilter();
  
  void learn();
  void saveSolution(std::string fname = "", MCsoln::Fmt fmt = MCsoln::BINARY );        ///< save projections to solnFile
  
  void params(SolverParams const& params) {m_params = params;} //set the parameters. 
  SolverParams& params(){return m_params;}  // allows changing the parameters directly. 
  SolverParams const& params() const {return m_params;}
  /** display info about filters. */
  void display();
  
 private:
  DP m_xy;
  SolverParams m_params; // make them a subclass? 
};
#endif // MCLEARNFILTER_H
