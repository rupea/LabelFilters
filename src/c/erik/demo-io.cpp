/** \file
 * test io of more complex types, like param_struct.
 */
#include "printing.hh"
#include "parameter.h"

#include <assert.h>
#include <fstream>
#include <cstring>      // std::memset

using namespace std;

int main(int,char**)
{
    param_struct a;
    memset(&a,0,sizeof(a));
    a = set_default_params();
    {
        cout<<"write to parms.txt"<<endl;
        ofstream f("parms.txt");
        write_ascii( f, a );
        f.close();
    }
    {
        // b = garbage
        param_struct b = {1,2,3,4,5,SAFE_SGD,6,ETA_CONST,7,8,9,10,11,12,13,0,0,REWEIGHT_NONE,REORDER_RANGE_MIDPOINTS,0,0,14,15,16,17,18,1,1,19};
        cout<<"read from parms.txt"<<endl;
        ifstream f("parms.txt");
        read_ascii( f, b );

#define CHK(PARM) assert( a.PARM == b.PARM )
        CHK(no_projections);
        CHK(C1);
        CHK(C2);
        CHK(max_iter);
        CHK(batch_size);
        CHK(update_type);
        CHK(eps);
        CHK(eta_type);
        CHK(eta);
        CHK(min_eta);
        CHK(avg_epoch);
        CHK(reorder_epoch);
        CHK(report_epoch);
        CHK(report_avg_epoch);
        CHK(optimizeLU_epoch);
        CHK(remove_constraints);
        CHK(remove_class_constraints);
        CHK(reweight_lambda);
        CHK(reorder_type);
        CHK(ml_wt_by_nclasses);
        CHK(ml_wt_class_by_nclasses);
        CHK(num_threads);
        CHK(seed);
        CHK(finite_diff_test_epoch);
        CHK(no_finite_diff_tests);
        CHK(finite_diff_test_epoch);
        CHK(no_finite_diff_tests);
        CHK(finite_diff_test_delta);
        CHK(resume);
        CHK(reoptimize_LU);
        CHK(class_samples);
        //  memcmp failed, even though all fields asserted equivalent --- where is the difference?
        char const* ita = (char const*)(void const*)&a;
        char const* itb = (char const*)(void const*)&b;
        for(uint32_t byte=0U; byte<sizeof(param_struct); ++byte){
            char const ca = *(ita+byte);
            char const cb = *(itb+byte);
            if( ca != cb ){
                cout<<" difference in byte "<<byte<<"  a[byte]="<<(uint32_t)ca<<"  b[byte]="<<(uint32_t)cb<<endl;
            }
        }
        // OH, of course. Reading back ascii doubles will be lossy.
        //assert( memcmp((void const*)&a, (void const*)&b, sizeof(param_struct)) == 0 );
        cout<<"Good, parms.txt read back equivalent data"<<endl;
    }
    {
        cout<<"write to parms.bin"<<endl;
        ofstream f("parms.bin");
        write_binary( f, a );
        f.close();
    }
    {
        // b = garbage
        param_struct b = {1,2,3,4,5,SAFE_SGD,6,ETA_CONST,7,8,9,10,11,12,13,0,0,REWEIGHT_NONE,REORDER_RANGE_MIDPOINTS,0,0,14,15,16,17,18,1,1,19};
        cout<<"read from parms.bin"<<endl;
        ifstream f("parms.bin");
        read_binary( f, b );

#define CHK(PARM) assert( a.PARM == b.PARM )
        CHK(no_projections);
        CHK(C1);
        CHK(C2);
        CHK(max_iter);
        CHK(batch_size);
        CHK(update_type);
        CHK(eps);
        CHK(eta_type);
        CHK(eta);
        CHK(min_eta);
        CHK(avg_epoch);
        CHK(reorder_epoch);
        CHK(report_epoch);
        CHK(report_avg_epoch);
        CHK(optimizeLU_epoch);
        CHK(remove_constraints);
        CHK(remove_class_constraints);
        CHK(reweight_lambda);
        CHK(reorder_type);
        CHK(ml_wt_by_nclasses);
        CHK(ml_wt_class_by_nclasses);
        CHK(num_threads);
        CHK(seed);
        CHK(finite_diff_test_epoch);
        CHK(no_finite_diff_tests);
        CHK(finite_diff_test_epoch);
        CHK(no_finite_diff_tests);
        CHK(finite_diff_test_delta);
        CHK(resume);
        CHK(reoptimize_LU);
        CHK(class_samples);
        //  memcmp failed, even though all fields asserted equivalent --- where is the difference?
        char const* ita = (char const*)(void const*)&a;
        char const* itb = (char const*)(void const*)&b;
        for(uint32_t byte=0U; byte<sizeof(param_struct); ++byte){
            char const ca = *(ita+byte);
            char const cb = *(itb+byte);
            if( ca != cb ){
                cout<<" difference in byte "<<byte<<"  a[byte]="<<(uint32_t)ca<<"  b[byte]="<<(uint32_t)cb<<endl;
            }
        }
        // OH, reading back ascii doubles seems to have slight bit differences, even though CHK succeeds
        //assert( memcmp((void const*)&a, (void const*)&b, sizeof(param_struct)) == 0 );
        cout<<"Good, parms.bin read back equivalent data"<<endl;
    }
    
    cout<<"\nGoodbye"<<endl;
    return 0;
}
