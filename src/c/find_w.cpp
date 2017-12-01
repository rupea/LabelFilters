
#include "find_w.hh"            // get template definition of the solve_optimization problem
#include "mcsolver.hh"          // template impl of MCsolver version
#include "normalize.h"          // MCxyData support funcs
#include "constants.h" // MCTHREADS
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;

// Explicitly instantiate templates into the library

// ----------------- Eigen native Dense and Sparse --------------
template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        const DenseM& x,                        // Dense
                        const SparseMb& y,
                        const param_struct& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        const SparseM& x,                       // Sparse
                        const SparseMb& y,
                        const param_struct& params);

// ---------------- External Memory variants --------------------
template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        //ExtConstDenseM& x,                    // type lookup --> no match
                        Eigen::Map<DenseM const> const& x,      // external-memory Dense
                        SparseMb const& y,
                        param_struct const& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        Eigen::MappedSparseMatrix<double, Eigen::RowMajor> const& x,// double const WON'T WORK
                        SparseMb const& y,
                        param_struct const& params);
                        




