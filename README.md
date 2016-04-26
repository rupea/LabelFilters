To compile: 

> cd src
> make

Eigen WILL NOT compile under modern g++ (g++-5.3, for example)
Please use g++-4.9.x or so.

Main function: src/octave/run_svm_test.m

- This branch is for milde support.
	> cd src/c
	> make all test && echo YAY
- the API has been cleaned up, C++11 is assumed.
- NO matlab/octave required
- NO liblinear
- It makes libmcfilter libraries
- and provides fully-C++ utilities showing how to:
  - mcgenx : generate test problems
  - mcsolve : solve for projections (from scratch or continue a previous soln)
  - mcdumpsoln : dump above .soln output files
  - mcproj : apply a .soln to test examples, outputting remaining possible classes.
- ... and a lot of test code, with lots of assertions in debug mode
- many bugfixes

- it does not work exactly like old code, but the internals of the solver are mostly
  the same.

- TODO: when within Milde:
  - remove historical cruft, all matlab, and all liblinear stuff
  - provide a repo --> eigen conversion utility for dense/sparse, slc/mlc data
    - Eigen matrices must be memory contiguous
    - y label are only supported as a integers 0..nClasses-1 (not strings as in repo)

TODO:
 - --update=SAFE should be the default
 - --update=SAFE should set the batch size to 1
 - --update=SAFE should have parameters to control the eta schedule
   - for example, "cautious" problems like mnist might use
     eta_bigger = 1.01 and eta_smaller = 0.8
 - trying out a change to keep --update=SAFE eta and avoid resetting it to "eta_t"
 - add s_rownormx to lua api, to verify update_safe_SGD normalization effects.

After seeing how poor mnist results were (without data augmentation)
I think we need to:
 - add setQuantiles support
 - add per-projection binary projection output
 - add per-projection quantile-based scoring output
   - and a 'final-score' output
 - consider mixture-of-experts scoring for binary/quantile projectors
