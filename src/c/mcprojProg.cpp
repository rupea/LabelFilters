
#include "mcprojProg.hpp"
#include "mcsolveProg.hpp"
#include "printing.hh"
#include "normalize.h"
#include "mcpredict.hh"
#include "predict.hh"           // low-level tests [optional]
#include "utils.h"              // OUTWIDE

//#include <omp.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace opt {
    using namespace std;
    /**
     * - What are the default parameters?
     *   - 1. if given explicitly as \c defparms
     *   - 2. if --solnFile, adopt parms from a previous run
     *   - 3. else use library defaults
     */
    MCprojProgram::MCprojProgram( int argc, char**argv
                                  , int const verbose/*=0*/ )
        : ::opt::MCprojArgs( argc, argv )
        , soln()
        , solveProg(nullptr)           // empty shared pointer
        , xy( new MCxyData() )
        , feasible()
        //, projFeasible()
    {
        // defaults: if given explicitly as \c defparms
        if(verbose>=1){
            //++A::verbose;
            cout<<" +MCprojProgram --xfile="<<A::xFile;
            if(A::solnFile.size()) cout<<" --solnFile="<<A::solnFile;
            if(A::outFile.size()) cout<<" --output="<<A::outFile;
            if(A::maxProj) cout<<" --proj="<<A::maxProj;
            if(A::outBinary) cout<<" -B";
            if(A::outText)   cout<<" -T";
            if(A::outSparse) cout<<" -S";
            if(A::outDense)  cout<<" -D";
            if(A::yFile.size()) cout<<(A::yPerProj?" --Yfile=":" --yfile")<<A::yFile;
            //if(A::threads)   cout<<" --threads="<<A::threads;
            if(A::xnorm)     cout<<" --xnorm";
            if(A::xunit)     cout<<" --xunit";
            if(A::xscale!=1.0) cout<<" --xscale="<<A::xscale;
            cout<<endl;
        }
    }
    MCprojProgram::MCprojProgram( MCsolveProgram * const solver, std::string mod/*=std::string()*/ )
        : ::opt::MCprojArgs( mod )
        , soln( solver->getSoln() )     // COPY the solver's solution  FIXME (soln doesn't arrive, can tryRead to reread it)
        , solveProg( solver )
        , xy( solver->xy )
        , feasible()
        //, projFeasible()
    {
        assert( xy );
        assert( solver );
        cout<<"+MCprojProgram: "<<xy->shortMsg()<<endl;
        cout<<"Initial MCprojProgram soln is "; soln.pretty(cout); cout<<endl;
    }
    MCprojProgram::~MCprojProgram(){
        if( solveProg && solveProg->projProg ){  // reset known shared_ptr to me [circular]
            //solveProg->projProg.reset();
            solveProg->projProg = nullptr;
        }
    }

    void MCprojProgram::tryRead( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1){cout<<" MCprojProgram::tryRead()"; cout.flush();}
        {
            try{
                ifstream sfs(A::solnFile);
                if(!sfs.good()) {cerr<<" solnfile "<<A::solnFile<<endl; throw std::runtime_error("trouble opening soln file");}
                soln.read( sfs );
                sfs.close();
            }catch(std::exception const& e){
                cerr<<"Error reading solnfile "<<A::solnFile<<endl<<e.what()<<endl;
                throw;
            }
        }
        if(verbose>=2) soln.pretty(cout);
        // obtain this->xy->MCxyData::{xDense, denseOk, xSparse,sparseOk, y}
        {
            if(verbose>=1) cout<<" try reading DenseM from xFile="<<xFile<<endl;
            xy->xread(xFile);    // tries binary dense, sparse, (others?)
        }
#if 1 // enable when sparse projection seems ok now... ???
        if(xnorm && xy->sparseOk){
            xy->xDense = DenseM( xy->xSparse );     // convert sparse --> dense
            xy->xSparse.resize(0,0);
            xy->sparseOk = false;
            xy->denseOk = true;
        }
#endif
        //SparseMb xy->y;                         // now this is optional
        // read SparseMb xy->y;
        if(A::yFile.size()){
            xy->yread(yFile);
            assert( xy->y.size() > 0 );
        }else{
            assert( xy->y.size() == 0 );
        }
#ifndef NDEBUG
        assert( xy->denseOk || xy->sparseOk );
        if( xy->y.size() ){ // xy->y is optional, for projection operation (allows validation)
            if( xy->denseOk ){
                assert( xy->xDense.rows() == xy->y.rows() );
            }else{ //xy->sparseOk
                assert( xy->xSparse.rows() == xy->y.rows() );
                // col-norm DISALLOWED
            }
        }
#endif
        if( xy->sparseOk && xnorm )
            throw std::runtime_error("sparse --xfile does not support --xnorm");
        if( xy->denseOk ){
            if( A::xnorm ){
                VectorXd xmean;
                VectorXd xstdev;
                if(verbose>=1){
                    cout<<"xy->xDense ORIG:\n"<<xy->xDense<<endl;
                    cout<<" xnorm!"<<endl;
                }
#if 1
                col_normalize(xy->xDense,xmean,xstdev);
                if(verbose>=1){
                    cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
                    cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
                }
#else
                normalize_col(xy->xDense);
#endif
            }
        }
        if( A::xunit ){
            xy->xrunit();
        }
        if( A::xscale != 1.0 ){
            xy-> xscale( A::xscale );
        }

        if(verbose>=2){
            if( xy->denseOk ){
                cout<<"--------- xy->xDense"<<prettyDims(xy->xDense)<<":\n";
                if(xy->xDense.size()<1000||verbose>=3) cout<<xy->xDense<<endl;
            }else{ //xy->sparseOk
                cout<<"--------- xy->xSparse"<<prettyDims(xy->xSparse)<<":\n";
                if(xy->xSparse.size()<1000||verbose>=3) cout<<xy->xSparse<<endl;
                //if( xnorm ){ cout<<" xnorm!"<<endl; col_normalize(xy->xSparse,xmean,xstdev); }
                // col-norm DISALLOWED
                if( xnorm ) throw std::runtime_error("sparse --xfile does not support --xnorm");
            }
            if( xy->y.size() <= 0 ) cout<<"xy->y: (no validation data)"<<endl;
            else{
                cout<<"--------- xy->y"<<prettyDims(xy->y)<<":\n";
                if(xy->y.size()<1000||verbose>=3) cout<<xy->y<<endl;
            }
        }
        //throw std::runtime_error("TBD");
    }
    /** x. Note: in practice on might instead want to <em>dig down</em> into
     * \c project(x,soln) and see how the statistics are doing after each single
     * [sequential] projection is applied.  Since classes can be very frequently
     * incorrectly eliminated, could the set of projection lines be used as a
     * mixture of experts instead?  Under what run conditions would that be a
     * correct view?
     *
     * \todo project --> mcpredict.h (project) --> mcpredict.cpp (projectionsToBitsets)
     *       modify projectionsToBitsets to track per-projection all-example stats
     *       (like ConfusionMatrix PER PROJECTION)
     */
    void MCprojProgram::tryProj( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1) cout<<" MCprojProgram::tryProj() "<<(xy->denseOk?"dense":xy->sparseOk?"sparse":"HUH?")<<endl;
#if NDEBUG
        { // This is still fairly high level -- it does ALL projections in x
            cout<<"\t\tjust for show test... START (deprecated)"<<endl;
            VectorXsz no_active;
            ActiveDataSet * ads=nullptr;
            if( xy->denseOk ){
                ads = getactive( no_active, xy->xDense,
                                 soln.weights_avg, soln.lower_bounds_avg, soln.upper_bounds_avg,
                                 /*verbose=*/true );
            }else{
                ads = getactive( no_active, xy->xSparse,
                                 soln.weights_avg, soln.lower_bounds_avg, soln.upper_bounds_avg,
                                 /*verbose=*/true );
            }
            assert( ads != nullptr );
            free_ActiveDataSet(ads); delete ads; ads=nullptr;
            cout<<"Good: got back from original 'getactive' call"<<endl;
            cout<<"\t\tjust for show test... DONE"<<endl;
        }
#endif
        if( A::yPerProj || A::maxProj )
        {
            cout<<(A::yPerProj?" -xy->y":" ")<<(A::maxProj?" -p":" ")<<" options **NOT IMPLEMENTED**"
                <<"\nHere, may want to unroll predict.cpp: projectionsToActiveSet per projection"
                <<endl;
        }if(1){ // high-level "do ALL projections" routine
            if( xy->denseOk ){
                feasible = project( xy->xDense, soln );
                if(verbose>=1){                                // dump x and 'feasible' classes
                    cout<<" feasible["<<feasible.size()<<"] classes, after project(xy->xDense,soln)"; cout.flush();
                    if(verbose>=2){
                        cout<<":"<<endl;
                        for(uint32_t i=0U; i<feasible.size(); ++i){
                            auto const& fi = feasible[i];
                            cout<<" x.row("<<setw(4)<<i<<")=";
                            OUTWIDE(cout,40,xy->xDense.row(i));
                            cout<<"   classes: ";
                            for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) cout<<" "<<c;
                            //for(uint32_t c=0U; c<fi.size(); ++c) cout<<fi[c];
                            cout<<endl;
                        }
                    }else{
                        cout<<endl;
                    }
                }
            }else if( xy->sparseOk ){
                if(A::xnorm && verbose>=1) cout<<"// normalization NOT YET SUPPORTED for sparse"<<endl;
                feasible = project( xy->xSparse, soln );
                if(verbose>=1){                                    // dump x and 'feasible' classes
                    cout<<" feasible["<<feasible.size()<<"] classes, after project(xy->xDense,soln)"; cout.flush();
                    if(verbose>=2){
                        cout<<":"<<endl;
                        for(uint32_t i=0U; i<feasible.size(); ++i){
                            auto const& fi = feasible[i];
                            cout<<" x.row("<<setw(4)<<i<<")=";
                            //OUTWIDE(cout,40,xy->xSparse.row(i)); // Eigen bug: unwanted CR (dense row output OK)
                            {
                                ostringstream oss;
                                oss<<xy->xSparse.row(i);
                                string s = oss.str();
                                cout<<setw(40)<<s.substr(0,s.size()-1);
                            }
                            cout<<"   classes: ";
                            for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) cout<<" "<<c;
                            cout<<endl;
                        }
                    }else{
                        cout<<endl;
                    }
                }
            }else{
                throw std::runtime_error("neither sparse nor dense training x was available");
            }
        }
    }
    struct ConfusionMatrix{
        size_t tp;              ///< true positives
        size_t fp;              ///< false positives
        size_t tn;              ///< true negatives
        size_t fn;              ///< false negatives
        // extensions ...
        size_t n;               ///< # of examples used
        static size_t const nhist = 20U;
        struct {
            size_t truu;        ///< count true with this number of labels
            size_t pred;        ///< count predictions with this number of labels
        } hist[nhist];
        ConfusionMatrix()
            : tp(0U), fp(0U), tn(0U), fn(0U), n(0U), hist{{0,0}}
        {}
        /// @name utility accessors
        //@{
        size_t trueLabels() const {return tp+fn;}
        size_t predLabels() const {return tp+fp;}
        double precision() const {return static_cast<double>(tp)/(tp+fp);}
        double recall() const {return static_cast<double>(tp)/(tp+fn);}
        double f1() const {return static_cast<double>(tp+tp)/(tp+tp+fp+fn);}
        /** hist[] true label distribution as fraction */
        double fracTrue( size_t const nlabels ){
            if( nlabels >= nhist ) throw std::runtime_error("hist[] array bound exceeded");
            return static_cast<double>(hist[nlabels].truu) / n; //trueLabels();
        }
        /** hist[] predicted label distribution as fraction */
        double fracPred( size_t const nlabels ){
            if( nlabels >= nhist ) throw std::runtime_error("hist[] array bound exceeded");
            return static_cast<double>(hist[nlabels].pred) / n; //predLabels();
        }
        //@}
    };
    static inline constexpr double round( double const x, double const unit=0.01 ){
        return static_cast<size_t>(x/unit+0.5) * unit;
    }
    static struct ConfusionMatrix confusion( std::vector<boost::dynamic_bitset<>> const& vbs
                                             , SparseMb y ){
        ConfusionMatrix cm;
        if( vbs.size() != y.rows() )
            throw std::runtime_error("confusion(vbs,y): #predicted != #labelled examples");
        cm.n = vbs.size();
        // overall confusion matrix counters:
        cm.tp = 0U, cm.fp = 0U, cm.tn = 0U, cm.fn = 0U;
        for(size_t i=0U; i<cm.n; ++i){
            auto const& fi = vbs[i];                    // predicted labels
            if( i < y.rows() ){
                size_t tp = 0U, fn=0U;
                for(SparseMb::InnerIterator it(y,i); it; ++it){
                    int const cls = it.col();           // true label
                    if( fi[cls] ) ++tp; else ++fn;
                }
                if( (tp+fn) < cm.nhist ) ++cm.hist[(tp+fn)].truu;
                size_t const predPositv = fi.count();      // slow operation
                if( predPositv < cm.nhist ) ++cm.hist[predPositv].pred;
                size_t const predNegatv = fi.size() - predPositv;
                size_t const fp = predPositv - tp;
                size_t const tn = predNegatv - fn;
                cm.tp+=tp; cm.fp+=fp; cm.tn+=tn; cm.fn+=fn;
            }
        }
        return cm;
    }
    /** perhaps get rid of the return value here, now that more
     * powerful stats are done during validate ? */
    static struct ConfusionMatrix dumpFeasible(std::ostream& os
                                               , std::vector<boost::dynamic_bitset<>> const& vbs
                                               , SparseMb y
                                               , bool denseFmt=false
                                               , size_t firstRows=0U)
    {
        size_t const nOut = std::min( vbs.size(), (firstRows?firstRows:vbs.size()));
        // overall confusion matrix counters:
        size_t xtp = 0U, xfp = 0U, xtn = 0U, xfn = 0U;
        for(uint32_t i=0U; i<nOut; ++i){
            auto const& fi = vbs[i];
            os<<i<<" ";
            if( denseFmt ){
                for(uint32_t c=0U; c<fi.size(); ++c) os<<fi[c];
            }else{
                for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) os<<" "<<c;
            }
            if( i < y.rows() ){
                size_t tp = 0U, fn=0U;
                os<<" true:";
                for(SparseMb::InnerIterator it(y,i); it; ++it){
                    int cls = it.col();
                    os<<" "<<cls;
                    if( fi[cls] ) ++tp; else ++fn;
                }
                size_t const npositive = fi.count();      // slow operation
                size_t const nnegative = fi.size() - npositive;
                size_t const fp = npositive - tp;
                size_t const tn = nnegative - fn;
                os<<"\t| tp="<<tp<<" fp="<<fp<<" tn="<<tn<<" fn="<<fn;
                xtp+=tp; xfp+=fp; xtn+=tn; xfn+=fn;
            }
            os<<endl;
        }
        if( y.size() )
            os  <<  "Overall confusion matrix: tp="<<setw(8)<<xtp<<" fp="<<setw(8)<<xfp
                <<"\n                          tn="<<setw(8)<<xtn<<" fn="<<setw(8)<<xfn<<endl;
        ConfusionMatrix r;
        r.tp=xtp; r.fp=xfp; r.tn=xtn; r.fn=xfn;
        return r;
    }
    void MCprojProgram::trySave( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if(verbose>=1) cout<<" MCprojProgram::trySave()"
            <<"\tSaving feasible "<<(outBinary?"Binary":"Text")<<" "
                <<(outSparse?"Sparse":"Dense")<<" classes --> "
                <<(outFile.size()? outFile: string("cout"))
                <<endl;
        if(outFile.size()==0){ // to cout, as Text
            if(verbose>=0 && outBinary) cout<<"## Warning: Binary flag ignored for cout"<<endl;
            cout<<"## solnFile: "<<A::solnFile<<endl;
            cout<<"##    xFile: "<<A::xFile<<endl;
            cout<<"## feasible["<<feasible.size()<<"x"<<soln.lower_bounds.rows()<<"] classes"
                ", after project(x,soln):"<<endl;
            dumpFeasible( cout, feasible, xy->y, A::outDense, /*firstRows=*/0U );
        }else{
            ofstream ofs(outFile);
            if( ! ofs.good() ) throw std::runtime_error("trouble opening outFile");
            if(outBinary){
                if(outSparse){
                    cout<<"## Warning: Sparse flag ignored -- see printing.hh to implement it 'magically'"<<endl;
                }
                detail::io_bin( ofs, feasible );
                if(verbose>=2) cout<<"Note:\tIn C++, use printing.hh\n\t\tio_bin(ifstream(\""<<outFile
                    <<"\"),vector<boost::dynamic_bitset<>>&)\n\tcan read the projections binary file";
            }else{ // outText
                ofs<<"## solnFile: "<<A::solnFile<<endl;
                ofs<<"##    xFile: "<<A::xFile<<endl;
                ofs<<"## feasible["<<feasible.size()<<"x"<<soln.lower_bounds.rows()<<"] classes"
                ", after project(x,soln):"<<endl;
                ConfusionMatrix cm = dumpFeasible( ofs, feasible, xy->y, A::outDense, /*firstRows=*/0U );
                if( xy->y.size() ){
                    cout<<"Overall confusion matrix: tp="<<setw(8)<<cm.tp<<" fp="<<setw(8)<<cm.fp
                        <<"\n                          tn="<<setw(8)<<cm.tn<<" fn="<<setw(8)<<cm.fn
                        <<"\n Average true predictions per example = "<<(cm.tp+cm.fp)/xy->y.rows()
                        <<endl;
                }
            }
            if( ! ofs.good() ) throw std::runtime_error("trouble writing outFile");
            ofs.close();
        }
    }
    void MCprojProgram::tryValidate( int const verb/*=0*/ )
    {
        int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
        if( A::yPerProj || A::maxProj )
        {
            cout<<(A::yPerProj?" -xy->y":" ")<<(A::maxProj?" -p":" ")<<" options **NOT IMPLEMENTED**"
                <<"\nHere, may want to unroll predict.cpp: projectionsToActiveSet per projection"
                <<endl;
        }
        if( xy->y.size() ){
            cout<<" McprojProgram::tryValidate() against --yfile "<<yFile<<endl;
            ConfusionMatrix cm = confusion( feasible, xy->y );
            cout<<"Overall confusion matrix: tp="<<setw(8)<<cm.tp<<" fp="<<setw(8)<<cm.fp
                <<"\n                          tn="<<setw(8)<<cm.tn<<" fn="<<setw(8)<<cm.fn
                <<"\n Average labels      per example = "
                <<setw(12)<<static_cast<double>(cm.tp+cm.fn)/xy->y.rows()
                <<"  hist%:";
            size_t iz;
            for(iz=cm.nhist; --iz<cm.nhist; ) if(cm.hist[iz].truu) break;
            ++iz;
            for(size_t i=0U; i<iz; ++i) cout<<setw(6)<<round(cm.fracTrue(i)*100.0,0.01);
            cout<<"\n Average predictions per example = "
                <<setw(12)<<static_cast<double>(cm.tp+cm.fp)/xy->y.rows()
                <<"  hist%:";
            for(iz=cm.nhist; --iz<cm.nhist; ) if(cm.hist[iz].pred) break;
            ++iz;
            for(size_t i=0U; i<iz; ++i) cout<<setw(6)<<round(cm.fracPred(i)*100.0,0.01);
            cout<<"\n Precision (tp/tp+fp)            = "
                <<setw(12)<<static_cast<double>(cm.tp)/(cm.tp+cm.fp)
                <<"\n Recall    (tp/tp+fn)            = "
                <<setw(12)<<static_cast<double>(cm.tp)/(cm.fp+cm.fn)
                <<"\n F1        (2tp/(2tp+fp+fn)      = "
                <<setw(12)<<static_cast<double>(cm.tp+cm.tp)/(cm.tp+cm.tp+cm.fp+cm.fn)
                <<endl;
        }else{
            if(verbose>=1) cout<<" MCprojProgram::tryValidate()  (no yFile)"
                <<endl;
        }
    }
}//opt::
