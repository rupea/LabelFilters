/** \file
 * learn projections using only C++ */
#include "../find_w.h" //old implementation
#include "../predict.h" //old implementation
#include "../mcsolver.h" //new implementation
#include "../mcfilter.h" //new implementation
#include "utils.h"              // labelVec2Mat
#include "Eigen/Dense"
#include "Eigen/Sparse"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <stdlib.h>
#include <fstream>
using namespace std;
using namespace Eigen;

#define TESTNUM 0
//#define PROBLEM 2U
param_struct params = set_default_params();
int testnum = TESTNUM;
bool use_mcsolver = true;
bool use_dense = true;
std::string saveBasename = std::string("");
//size_t problem = 2U;
string problem_msg("random data");
int verbose = 0;

// parse arguments
int testparms(int argc, char**argv, param_struct & params) {
    //problem = PROBLEM;
    testnum = TESTNUM;
    use_mcsolver = true;
    use_dense = true;
    bool dohelp=false;
    for(int a=1; a<argc; ++a){
        for(char* ca=argv[a]; *ca!='\0'; ){
            if(*ca == 'h'){
                cout<<" Options:"
                    //<<"\n   pN          test problem N in [0,2] w/ base parameter settings [default=2]"
                    <<"\n   0-9         parameter modifiers, [default=0, no change from pN defaults]"
                    <<"\n   m|o         m[default] mcsolver code | o original code"
                    <<"\n   D|S         Dense[default] | Sparse matrices"
                    <<"\n   s<string>   save file base name (only for 'm') [default=none]"
                    <<"\n   h           this help"
                    <<"\n   v           increase verbosity [default=0]"
                    <<"\n* s<string> can only be the final option"
                    <<"\n* default runs Dense MCsolver code"
                    <<endl;
                dohelp=true;
            }else if(isdigit(*ca)){
                testnum = 0U;
                for( ; isdigit(*ca); ++ca) { testnum=10U*testnum + (*ca - '0'); }
                goto HaveNextCA;
            }else if(*ca == 'v')         ++verbose;
            else if(*ca == 'm')         use_mcsolver=true;
            else if(*ca == 'o')         use_mcsolver=false;
            else if(*ca == 'D')         use_dense=true;
            else if(*ca == 'S')         use_dense=false;
            //else if(*ca == 'p')         problem = *++ca - '0';
            else if(*ca == 's'){
                saveBasename.assign(++ca);
                break;
            }
            ++ca;
HaveNextCA:
            continue;
        }
    }
    //if(problem>2U) problem=2U;
    if(dohelp){
        cout<<" use_mcsolver="<<use_mcsolver<<" use_dense="<<use_dense
            <<" testnum="<<testnum<<" saveBasename='"<<saveBasename<<"'"<<endl;
        exit(0);
    }
    return testnum;
}

// apply 0-9 parameter modifiers
void apply_testnum(param_struct & params)
{
    cout<<" demo-proj running testnum "<<testnum;
    switch( testnum ){
      case(0):
          cout<<" all parameters default";
          break;
      case(1):
          params.optimizeLU_epoch = 0;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch;
          break;
      case(2):                // mcsolver does not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at end of first dim (after t=100000)
          params.avg_epoch = 1200;    
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(3):        // make nAccSortlu_avg increment (for sure)
          params.optimizeLU_epoch = 0;
          params.avg_epoch = 16000;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(4):                // original annd mcsolver BOTH do not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at t=1400
          params.avg_epoch = 1200;
          params.report_avg_epoch = 1400;
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(5):
          params.reorder_type = REORDER_PROJ_MEANS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(6):
          params.reorder_type = REORDER_RANGE_MIDPOINTS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(7):        // make nAccSortlu_avg increment (for sure)
          params.reorder_epoch = 1000;   // default
          params.optimizeLU_epoch = 0;
          params.report_avg_epoch = 9999;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(8):        // make nAccSortlu_avg increment (for sure)
          params.reorder_epoch = 1000;   // default
          params.optimizeLU_epoch = 0;
          params.report_avg_epoch = params.report_epoch*2U;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      default:
          cout<<" OHOH! testnum = "<<testnum<<", really?"<<endl;
          throw std::runtime_error(" UNKNOWN testnum (running defaults)");
    }
    cout<<endl;
}

void mcSave(std::string saveBasename, MCsoln const& soln){            
    if( saveBasename.size() > 0U ){
        string saveTxt(saveBasename); saveTxt.append(".soln");
        cout<<" Saving to file "<<saveTxt<<endl;
        try{
            ofstream ofs(saveTxt);
            soln.write( ofs, MCsoln::TEXT );
            ofs.close();
        }catch(std::exception const& e){
            cout<<"OHOH! Error during text write of demo.soln "<<e.what()<<endl;
            throw(e);
        }
        string saveBin(saveBasename); saveBin.append("-bin.soln");
        cout<<" Saving to file "<<saveBin<<endl;
        try{
            ofstream ofs(saveBin);
            soln.write( ofs, MCsoln::BINARY );
            ofs.close();
        }catch(std::exception const& what){
            cout<<"OHOH! Error during binary write of demo.soln"<<endl;
            throw(what);
        }
    }
}

int main(int argc,char** argv)
{

#ifdef _OPENMP
    cout<<" _OPENMP defined, omp_get_max_threads ... "<<omp_get_max_threads()<<endl;
#endif

    //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
    //  VectorXd y(10000),objective_val;
    params = set_default_params();
    testparms(argc,argv, params);
    params.no_projections = 4U;
    params.max_iter=100000U;              // default 1e6 takes 10-15 minutes
    apply_testnum(params);
    //    params.report_epoch = 1000;
    //    params.verbose = 0;
    int const rand_seed = 117;
    int const p = params.no_projections;
    //int const d = 12345U;       // x training data dimensionality ~ 440 s runtime
    int const d = 281U;           // ~ 30 s run
    int const k = 5U;             // number of classes

    // x training data and y class labels
    srand(rand_seed);
    DenseM x(281,d);
    VectorXd yVec(281);
    x.setRandom();
    srand(rand_seed);
    cout<<" rand seed = "<<rand_seed<<" ,  x(100,100) = "<<x(100,100)<<endl;
    for (int i = 0; i < yVec.size(); i++) {
        //      y(i) = (i%1000)+1;
        yVec(i) = (i%k)+1;
    }
    SparseMb y = labelVec2Mat(yVec);

    if( use_dense ){
      //        if( ! use_mcsolver ){
      cout<<" *** BEGIN SOLUTION *** original code: "<<problem_msg<<endl;
      srand(rand_seed);
      // In fact, don't need to size or initialize
      cout<<" demo-proj.cpp, easy-init, dense "<<endl;
      DenseM weights, lower_bounds, upper_bounds;
      DenseM w_avg, l_avg, u_avg;
      VectorXd objective_val, o_avg;
      cout<<"  pre-run call to rand() returns "<<rand()<<endl;
      // these calls are important so that the compiler instantiates the right templates
      solve_optimization(weights,lower_bounds,upper_bounds,objective_val
			 ,w_avg,l_avg,u_avg,o_avg
			 ,x,y,params);
      VectorXsz no_active;
      ActiveDataSet* active_old=getactive(no_active, x, w_avg, l_avg, u_avg);
      //        }else{
      cout<<" *** BEGIN SOLUTION *** MCsolver code "<<problem_msg<<endl;
      srand(rand_seed);
      cout<<"  pre-run call to rand() returns "<<rand()<<endl;
      MCsolver mc;
      mc.solve( x, y, &params );
      MCsoln const& soln = mc.getSoln();
      mcSave(saveBasename, soln);
      MCfilter mf(soln);
      // check that the active classes are the same
      std::vector<boost::dynamic_bitset<>> active_mc;
      mf.filter(active_mc,x);	    
      for (int i =0; i<active_mc.size(); i++)
	{
	  if (active_mc[i] != *(active_old->at(i)))
	    {
	      cout << "Active sets do not match" << endl;
	      cout << "old one " << *(active_old->at(i)) << endl;
	      cout << "new one " << active_mc[i] << endl;
	      exit(-1);
	    }
	}	  
      if (saveBasename.size() > 0U){
	//test reading from solution file
	string saveTxt(saveBasename); saveTxt.append(".soln");	  
	MCfilter mf1; mf1.read(saveTxt);
	string saveBin(saveBasename); saveBin.append("-bin.soln");	  
	MCfilter mf2; mf2.read(saveBin);
	
	mf1.filter(active_mc,x);	    
	for (int i =0; i<active_mc.size(); i++)
	  {
	    if (active_mc[i] != *(active_old->at(i)))
	      {
		cout << "Active sets do not match (read from text)" << endl;
		cout << "old one " << *(active_old->at(i)) << endl;
		cout << "new one " << active_mc[i] << endl;
		exit(-1);
	      }
	  }	  
	mf2.filter(active_mc,x);	    
	for (int i =0; i<active_mc.size(); i++)
	  {
	    if (active_mc[i] != *(active_old->at(i)))
	      {
		cout << "Active sets do not match (read from binary)" << endl;
		cout << "old one " << *(active_old->at(i)) << endl;
		cout << "new one " << active_mc[i] << endl;
		exit(-1);
	      }
	  }
      }     
      //    }

    }else{        // sparse x
        // let's let the sparse dim be 4x larger, ?? and add one high dim of noise
        SparseM xs(x.rows(), 4*x.cols());
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(4*x.rows()*x.cols());
        for(int r=0; r<x.rows(); ++r){
            for(int c=0; c<x.cols(); ++c)
                tripletList.push_back(T(r,c,x(r,c)));
            //tripletList.push_back(T(r,int(c + (double(rand())/RAND_MAX)*2.5*c),double(rand()/RAND_MAX)));
        }
        xs.setFromTriplets(tripletList.begin(),tripletList.end());
	//        if( ! use_mcsolver ){
	cout<<" *** BEGIN SOLUTION *** original code: "<<problem_msg<<endl;
	// In fact, don't need to size or initialize
	srand(rand_seed);
	cout<<" demo-proj.cpp, easy-init, dense "<<endl;
	DenseM weights, lower_bounds, upper_bounds;
	DenseM w_avg, l_avg, u_avg;
	VectorXd objective_val, o_avg;
	cout<<"  pre-run call to rand() returns "<<rand()<<endl;
	// these calls are important so that the compiler instantiates the right templates
	solve_optimization(weights,lower_bounds,upper_bounds,objective_val
			   ,w_avg,l_avg,u_avg,o_avg
			   ,xs,y,params);
	VectorXsz no_active;
	ActiveDataSet* active_old=getactive(no_active, xs, w_avg, l_avg, u_avg);
	    
	//} else {
	
	cout<<" *** BEGIN SOLUTION *** MCsolver code "<<problem_msg<<endl;
	srand(rand_seed);
	cout<<"  pre-run call to rand() returns "<<rand()<<endl;
	MCsolver mc;
	mc.solve( xs, y, &params );
	MCsoln const& soln = mc.getSoln();
	mcSave(saveBasename, soln);	  
	MCfilter mf(soln);
	// check that the active classes are the same
	std::vector<boost::dynamic_bitset<>> active_mc;
	mf.filter(active_mc,xs);	    
	for (int i =0; i<active_mc.size(); i++)
	  {
	    if (active_mc[i] != *(active_old->at(i)))
	      {
		cout << "Active sets do not match" << endl;
		cout << "old one " << *(active_old->at(i)) << endl;
		cout << "new one " << active_mc[i] << endl;
		exit(-1);
	      }
	  }	  
	if (saveBasename.size() > 0U){
	  //test reading from solution file
	  string saveTxt(saveBasename); saveTxt.append(".soln");	  
	  MCfilter mf1; mf1.read(saveTxt);
	  string saveBin(saveBasename); saveBin.append("-bin.soln");	  
	  MCfilter mf2; mf2.read(saveBin);
	  
	  mf1.filter(active_mc,xs);	    
	  for (int i =0; i<active_mc.size(); i++)
	    {
	      if (active_mc[i] != *(active_old->at(i)))
		{
		  cout << "Active sets do not match (read from text)" << endl;
		  cout << "old one " << *(active_old->at(i)) << endl;
		  cout << "new one " << active_mc[i] << endl;
		    exit(-1);
		}
	    }	  
	  mf2.filter(active_mc,xs);	    
	  for (int i =0; i<active_mc.size(); i++)
	      {
		if (active_mc[i] != *(active_old->at(i)))
		  {
		    cout << "Active sets do not match (read from binary)" << endl;
		    cout << "old one " << *(active_old->at(i)) << endl;
		    cout << "new one " << active_mc[i] << endl;
		    exit(-1);
		  }
	      }
	}
	//}
    }


    cout<<" post-run call to rand() returns "<<rand()<<endl;
#if 0
    // sparse case
    SparseM xs = x.sparseView();
    //solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,params);
    xs.conservativeResize(281,1123497);
    DenseM sweights (1123497,1);
    sweights.setRandom();
    solve_optimization(sweights,lower_bounds,upper_bounds,objective_val,xs,y,params);
#endif

}

