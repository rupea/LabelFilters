The code implements the Label Filter algorithm for speeding up prediction in Extreme Classification problems (i.e. multi-label classification problems with an extremely large labels set). Label Filters are a computationally efficient technique for pre-selecting a small set of candidate labels for each test example before applying a more expensive multi-label classifier. For details please see the following paper:

Alexandru Niculescu-Mizil, Ehsan Abbasnejad: "[Label Filters for Large Scale Multilabel Classification](http://www.niculescu-mizil.org/paper.php?p=mcfilter.pdf)" - Proceedings of the 20th International Conference on AI and Statistics (AISTATS `17).


## Installation

### Dependencies
The code requires the following libraries:

- [boost](https://www.boost.org) for Dynamic Bitset and Program Options libraries
- [gperftools](https://github.com/gperftools/gperftools) for tcmalloc and profiler
- [Eigen](http://eigen.tuxfamily.org)  (included as a submodule)
- [CRoaring](http://roaringbitmap.org)  (included as a submodule)


### Compiling

First modify src/Makefile to point to the correct path for the boost libraries then execute:

```
cd src
make CRoaring #if using the included submodule
make
```

This will compile the label filter library and put the result into the bin/ directory

The compilation will produce the following files:

- libmcfilter.so -- label filter library, for dynamic linking
- libmcfilter.a -- label filter library, for static linking
- mcsolve -- example program to learn a filter
- mcpoj -- example program to apply the filter with linear classifiers.


## Using the code

```
cd Mediamill

gunzip train_split1_Mediamill_data.txt.gz test_split1_Mediamill_data.txt.gz

# learn the label filters
../bin/mcsolve -o mediamill_C1_0.1_C2_0.1.filter --C1=0.1 --C2=0.1 --nfilters=2 -x train_split1_Mediamill_data.txt --maxiter=1000000

# classify using an SVM model
numactl --interleave=all -- ../bin/mcproj --nProj={0,2} -f mediamill_C1_0.1_C2_0.1.filter -x test_split1_Mediamill_data.txt --modelFiles svm_C10_split1_Mediamil.svmmodel

# list all options
../bin/mcsolve --help
../bin/mcproj --help

```



## File formats

### Data file format

The data file uses the XML format.

First line is an optional header line:

```
nExamples nFeatures nClasses
```

Subsequently the file should contain one datapoint per line, in the following sparse format:

```
label1,label2,label3  feature:value feature:value ... feature:value  #comment
```

Labels are consecutive integers from 0 to nClasses-1.
Features are consecutive integers from 0 to nFeatures-1.

The code also supports a binary format with both dense and sparse storage.


### Label filter file format

The label filter file starts with a header row:

```
nFeatures nFilters nClasses
```

Then the parameters of the label filters are stored in three space separated matrices:

1. The filter directions as a `nFilters x d` matrix
2. The lower bounds as a `nFilters x nClass` matrix
3. The upper bounds as a `nFilters x nClass` matrix



### Linear model file format

The linear classifiers are stored in an SVMLight like format.

The first row in the file is an optional header:

```
nClasses nFeatures
```

Next the weights of each classifier are stored, one classifier per line, in the order of the labels (i.e. classifier corresponding to label '0' is first, then the classifier corresponding to label '1' and so on. The format is:

```
intercept feature:weight feature:weight ... feature:weight
```

Features are consecutive integers from 0 to nFeatures-1.

The intercept is optional. If not present, it is treated as 0.

The code also supports a binary format with both dense and sparse storage.


## Important classes

- MClearnFilter -- main class for training the label filters from data.
- MClinearClass -- main class for testing using a linear model and optional label filters. Handles prediction and evaluation.

- MCxyDataArgs -- defines parameters and command line options for handling the data
- MCsolveArgs -- defines parameters and command line options for learning the label filter
- MCprojectorArgs -- defines parameters and command line options for applying the label filters
- MCclassifierArgs -- defines parameters and command line options for applying the classifier

- MCxyData -- encapsulates the data for training/testing. Handles both dense and sparse data.  Manages loading and
saving data in various formats.
- MCsoln -- Encapsulates the label filter parameters. Manages loading and saving label filters in text or binary formats.
- PredictionSet -- encapsulates the predictions of the model.
- linearModel -- encapsulates a linear classifier. Handles loading, saving and prediction (not training)
