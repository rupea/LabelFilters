#include "normalize.h"
#include "typedefs.h"
#include <assert.h>

#include <iostream>
using namespace std;

template<typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,
              const Eigen::DenseBase<DerivedB>& b,
              const typename DerivedA::RealScalar& rtol
                  = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar& atol
                  = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
{
  return ((a.derived() - b.derived()).array().abs()
          <= (atol + rtol * b.derived().array().abs())).all();
}

int main(int,char**){
    int const verbose=2;
    uint32_t nErr=0U, nTest=0U;
    std::ostringstream errMsgs;
    {
        char const* test="3x3 col_normalize(x,m,s)";
        DenseM x(3,3);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (c+1)*(c+1)*(r+1);
        // Input: x
        //  1  4  9
        //  2  8 18
        //  3 12 27
        // Output: means
        // 2 8 18
        // Output: stdev
        // 1 4 9
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        col_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" col_normalize(x,m,s)\n"<<x<<endl;
        
        DenseM answer(3,3); answer<<-1,-1,-1, 0,0,0, 1,1,1;
        VectorXd mu(3);     mu<<2,8,18;
        VectorXd st(3);     st<<1,4,9;
        if(verbose>1)cout<<" answer\n"<<answer<<endl;
        if( allclose(x,answer) && allclose(m,mu) && allclose(s,st) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="3x4 col_normalize(x,m,s)";
        DenseM x(3,4);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = r*c;
        // Input: x
        // 0 0 0 0
        // 0 1 2 3
        // 0 2 4 6
        // Output: means
        // 0 1 2
        // Output: stdev (note stdev[0]==0)
        // 0 1 2
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        col_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" col_normalize(x,m,s)\n"<<x<<endl;
        DenseM answer(3,4); answer<<0,-1,-1,-1,   0,0,0,0,   0,1,1,1;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="2x3 col_normalize(x,m,s)";
        DenseM x(2,3);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (c+1)*(c+1)*(r+1);
        // Input: x
        //  1  4  9
        //  2  8 18
        // Output: means
        // 1.5 6 13.5
        // Output: stdev
        // 0.707 2.83 6.36
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        col_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" col_normalize(x,m,s)\n"<<x<<endl;
        double a=sqrt(0.5);
        DenseM answer(2,3); answer<<-a,-a,-a,  a,a,a;
        if(verbose>1)cout<<" answer\n"<<answer<<endl;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="1x3 col_normalize(x,m,s)";
        DenseM x(1,3);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (c+1)*(c+1)*(r+1);
        // Input: x
        //  1  4  9
        // Output: means
        // 1 4 9
        // Output: stdev
        // 0 0 0
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        col_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" col_normalize(x,m,s)\n"<<x<<endl;
        //double a=sqrt(0.5);
        DenseM answer(1,3); answer<<0,0,0;
        if(verbose>1)cout<<" answer\n"<<answer<<endl;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="3x3 row_normalize(x,m,s)";
        DenseM x(3,3);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (c+1)*(r+1);
        // Input: x
        // 1 2 3
        // 2 4 6
        // 3 6 9
        // Output: means
        // 2 4 6
        // Output: stdev
        // 1 2 3
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        row_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" row_normalize(x,m,s)\n"<<x<<endl;
        DenseM answer(3,3); answer<<-1,0,1,  -1,0,1,  -1,0,1;
        //cout<<" answer\n"<<answer<<endl;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="4x3 row_normalize(x,m,s)";
        DenseM x(4,3);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = r*c;
        // Input: x
        // 0 0 0
        // 0 1 2
        // 0 2 4
        // 0 3 6
        // Output: row means
        // 0 1 2 3
        // Output: stdev (note stdev[0]==0)
        // 0 1 2 3
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        row_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" row_normalize(x,m,s)\n"<<x<<endl;
        DenseM answer(4,3); answer<<0,0,0,  -1,0,1,  -1,0,1,  -1,0,1;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="3x2 row_normalize(x,m,s)";
        DenseM x(3,2);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = r*c*2;
        // Input: x
        // 0 0
        // 0 2
        // 0 4
        // Output: row means
        // 0 1 2
        // Output: stdev (note stdev[0]==0)
        // 0 sqrt(2)=0.707 sqrt(8)=2.83
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        row_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" row_normalize(x,m,s)\n"<<x<<endl;
        double a=sqrt(0.5);
        DenseM answer(3,2); answer<<0,0,  -a,a,  -a,a;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="3x1 row_normalize(x,m,s)";
        DenseM x(3,1);
        VectorXd m, s;
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = r;
        // Input: x
        // 0
        // 1
        // 2
        // Output: row means
        // 0 1 2
        // Output: stdev (note stdev[0]==0)
        // 0 0 0
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        row_normalize(x,m,s);
        if(verbose>1)cout<<" m="<<m.transpose()<<endl;
        if(verbose>1)cout<<" s="<<s.transpose()<<endl;
        if(verbose>0)cout<<" row_normalize(x,m,s)\n"<<x<<endl;
        DenseM answer(3,1); answer<<0,0,0;
        if( allclose(x,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    //
    // ----------------------- SparseM tests -------------------
    //
    {
        char const* test="3x3 normalize_row_remove_mean(SPARSE x)";
        SparseM x(3,3);
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = c+(c+1)*(r+1);
        // Input: x
        // 1 3 5
        // 2 5 8
        // 3 7 11
        // v=2 3 4     v^{-1} = 1/2 1/3 1/4
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        x.makeCompressed();
        normalize_row_remove_mean(x); //,m,s);
        if(verbose>0)cout<<" normalize_row_remove_mean(x)\n"<<x<<endl;
        
        DenseM calc=x;  // convert to dense
        DenseM answer(3,3); answer<<-1,0,1,  -1,0,1,  -1,0,1;
        if(verbose>1)cout<<" answer\n"<<answer<<endl;
        if( allclose(calc,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
    {
        char const* test="3x3 normalize_row(SPARSE x)";
        SparseM x(3,3);
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (r==0? 0: c==0? 0: r*(c+2));
        // Input: x
        // 0 0 0
        // 0 3 4
        // 0 6 8
        // 
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        x.makeCompressed();
        normalize_row(x); //,m,s);
        if(verbose>0)cout<<" normalize_row(x)\n"<<x<<endl;
        
        DenseM calc=x;  // convert to dense
        DenseM answer(3,3); answer<<0,0,0,  0,.6,.8,  0,.6,.8;
        if(verbose>1)cout<<" answer\n"<<answer<<endl;
        if( allclose(calc,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
#if 0 // not implemented -- throws if ever called
    {
        char const* test="3x3 normalize_row(SPARSE x)";
        SparseM x(3,3);
        for(int r=0; r<x.rows(); ++r)
            for(int c=0; c<x.cols(); ++c)
                x.coeffRef(r,c) = (c+1)*(r+1);
        // Input: x
        // 1 2 3
        // 2 4 6
        // 3 6 9
        if(verbose>0)cout<<" TEST "<<test<<" x\n"<<x<<endl;
        x.makeCompressed();
        normalize_row(x);
        if(verbose>0)cout<<" normalize_row(x)\n"<<x<<endl;
        DenseM calc = x;
        if(verbose>0)cout<<" dense : calc = x)\n"<<calc<<endl;
        DenseM answer(3,3); answer<<-1,0,1,  -1,0,1,  -1,0,1;
        //cout<<" answer\n"<<answer<<endl;
        if( allclose(calc,answer) ){
            ++nTest;
            if(verbose>1)cout<<test<<" GOOD"<<endl;
        }else{
            ++nErr;
            cout<<test<<" FAILED"<<endl;
            errMsgs<<test<<" FAILED"<<endl;
        }
    }
#endif
    if( nErr ){
        cout<<"\n Errors were encountered:"<<endl;
        cout<<errMsgs.str();
    }
    cout<<"\nPASSED "<<nTest-nErr<<" FAILED "<<nErr<<" tests"<<endl;
    return (nErr==0? 0: -1);
}
