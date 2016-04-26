#ifndef EVAL_W_H

#include "find_w.h"

/** \file
 * Here we use the SAME x and y format solver MCsoln, and apply {w,l,u}_avg to a test set.
 * - find_w.h uses just x and y as input and outputs an MCsoln (or .soln file)
 *
 * - original API was in evaluate_projection.h, evaluate.h, evaluate.hh ...
 * - We DO NOT confuse issue of whether x must first be transformed by some
 *   SVM projection step.  Our input is just the same 'x' that we used for the MCsoln.
 *
 * - In particular, note that \c predict.hh just iterates over examples,
 *   - i) transforming each example into a different space using DotProductInnerVector,
 *   - ii) and then working with
 * - So consider the ovaW (liblinear xform) to be just some arbitrary xform,
 *     then ignores
 *
 * - consider whether eval belongs inside MCsolver ?
 */

/** Evaluate a test set.
 * \template EIGENTYPE type of x matrix, outervector is training examples
 * \template EVALUATOR takes a single row (outervector) of x, a class k,
 * and returns a number increasing with likelihood of x being of class k.
 *
 * Example: EVALUATOR could form dot(x,svmW), where svmW is the one-vs-all
 *          separator for class K.
 * Example: EVALUATOR could return 1/dist(x,k) where distance is evaluated
 *          on the basisi of components on projected (or random) vectors
 *          to answer "x is close to centroid of class k"
 */
template< typename EIGENTYPE, typename EVALUATOR >
void eval( EIGENTYPE const& x, SparseMb& y, MCsoln const& sol, EVALUATOR const&f ); 
        
#define EVAL_W_H
#endif EVAL_W_H
