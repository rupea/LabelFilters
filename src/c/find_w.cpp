
#include "find_w.hh"            // get template definition of the solve_optimization problem


// Explicitly instantiate templates into the library

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds,
                        VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg,
                        VectorXd& objective_val_avg,
                        const DenseM& x,                // <-------- EigenType
                        const SparseMb& y,
                        const param_struct& params);

template
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds,
                        VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg,
                        VectorXd& objective_val_avg,
                        const SparseM& x,                // <-------- EigenType
                        const SparseMb& y,
                        const param_struct& params);
