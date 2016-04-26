
#include "typedefs.h"

#include <iostream>
#include <iomanip>
using namespace std;

/** convert x to new matrix adding quadratic dimensions to each InnerIterator (row).
 * Optionally scale the quadratic elements to keep them to reasonable range.
 * For example if original dimension is 0..N, perhaps scale the quadratic
 * terms by 1.0/N. */
void addQuadratic( SparseM & x, double const qscal=1.0 ){
    x.makeCompressed();
    VectorXi xsz(x.outerSize());
    for(int i=0U; i<x.outerSize(); ++i){
        xsz[i] = x.outerIndexPtr()[i+1]-x.outerIndexPtr()[i];
    }
    // final inner dim increases by square of inner dim
    SparseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
    // calc exact nnz elements for each row of q
    VectorXi qsz(x.outerSize());
    for(int i=0; i<x.outerSize(); ++i){
        qsz[i] = xsz[i] + xsz[i]*xsz[i];
    }
    q.reserve( qsz );           // reserve exact per-row space needed
    // fill q
    for(int r=0; r<x.outerSize(); ++r){
        for(SparseM::InnerIterator i(x,r); i; ++i){
            q.insert(r,i.col()) = i.value();    // copy the original dimension
            for(SparseM::InnerIterator j(x,r); j; ++j){
                int col = x.innerSize() + i.col()*x.innerSize() + j.col(); // x.innerSize() is 4
                q.insert(r,col) = i.value()*j.value()*qscal;  // fill in quad dims
            }
        }
    }
    q.makeCompressed();
    x.swap(q);
}
void addQuadratic( DenseM & x, double const qscal=1.0 ){
    DenseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
    for(int r=0; r<x.outerSize(); ++r){
        for(int i=0; i<x.innerSize(); ++i){
            q.coeffRef(r,i) = x.coeff(r,i);
            for(int j=0; j<x.innerSize(); ++j){
                int col = x.innerSize() + i*x.innerSize() + j;
                q.coeffRef(r,col) = x.coeff(r,i) * x.coeff(r,j) * qscal;
            }
        }
    }
    x.swap(q);
}
int main(int,char**){
    {
        cout<<" Sparse demo"<<endl;
        SparseM x(8,4);
        x.reserve(VectorXi::Constant(8,2));
        x.insert(1,0)=2.0;
        x.insert(2,1)=2.0;
        x.insert(3,2)=2.0;
        x.insert(4,3)=2.0;
        x.insert(5,0)=2.0;
        x.insert(5,1)=2.0;
        x.insert(6,0)=2.0;
        x.insert(6,2)=2.0;
        x.insert(7,1)=2.0;
        x.insert(7,2)=2.0;
        x.makeCompressed();
        cout<<" x sparse\n"<<x<<endl;
        cout<<" x.innerSize() = "<<x.innerSize()<<endl;
        cout<<" x.outerSize() = "<<x.outerSize()<<endl;

        cout<<" xsz? "; cout.flush();
        VectorXi xsz(x.outerSize());
        for(int i=0U; i<x.outerSize(); ++i){
            xsz[i] = x.outerIndexPtr()[i+1]-x.outerIndexPtr()[i];
            cout<<" "<<xsz[i]; cout.flush();
        }
        cout<<endl;

        SparseM q(8,4+4*4);
        VectorXi qsz(x.outerSize());
        for(int i=0; i<x.outerSize(); ++i){
            qsz[i] = xsz[i] + xsz[i]*xsz[i];
        }
        q.reserve( qsz );

        for(int r=0; r<x.outerSize(); ++r){
            for(SparseM::InnerIterator it(x,r); it; ++it){
                q.insert(r,it.col()) = it.value();  // copy the original dimension
            }
        }
        cout<<" q (copy x)\n"<<q<<endl;
        for(int r=0; r<x.outerSize(); ++r){
            for(SparseM::InnerIterator i(x,r); i; ++i){
                for(SparseM::InnerIterator j(x,r); j; ++j){
                    int col = x.innerSize() + i.col()*x.innerSize() + j.col(); // x.innerSize() is 4
                    //cout<<" i."<<i.col()<<" j."<<j.col()<<" xsz."<<xsz[r]<<" col."<<col<<endl;
                    q.insert(r,col) = i.value()*j.value();  // fill in quad dims
                }
            }
        }
        q.makeCompressed();
        cout<<" q (add quadratic dims)\n"<<q<<endl;

        addQuadratic( x );
        cout<<" addQuadratic(x) ->\n"<<x<<endl;
    }
    {
        cout<<"Dense demo"<<endl;
        DenseM x(8,4);
        x.setZero();
        x(1,0)=3.0;
        x(2,1)=3.0;
        x(3,2)=3.0;
        x(4,3)=3.0;
        x(5,0)=3.0;
        x(5,1)=3.0;
        x(6,0)=3.0;
        x(6,2)=3.0;
        x(7,1)=3.0;
        x(7,2)=3.0;
        cout<<" x dense\n"<<x<<endl;
        addQuadratic(x);
        cout<<" addQuadratic(DenseM)\n"<<x<<endl;
    }
    cout<<"\nGoodbye"<<endl;
}
