#ifndef __CONSTANTS_H
#define __CONSTANTS_H

// ***********  Constant values used
// most of these should be passed as arguments
const unsigned int OPT_MAX_ITER = 1e5; 	// Maximum number of iterations
const int OPT_MAX_REORDERING = 20; // maximum time the ordering of switches have to be changed
const double OPT_EPSILON = 1e-4; // optimization epsilon: how different the update for w is
const int PRINT_T = 0;                 	// print values in each iteration
const int PRINT_O = 1;                // print objective function in each epoch
const int PRINT_M = 0;                 	// print matrix operations
const int PRINT_I = 0;                 	// print Conditions in each iteration
const int STOCHASTIC_BATCH_SIZE = 100; // perform stochastic gradient with this batchsize; if -1 then complete GD
const int OPT_EPOCH = 1000;
// ********************************

#endif
