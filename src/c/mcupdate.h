
#include "mcsolver.hh"

struct MCupdate
{
    template<typename EigenType> 
        static void update( WeightVector& w, /*VectorXd& sortedLU, VectorXd& sortedLU_avg,*/
                            MCpermState & luPerm,      // sortlu and sortlu_avg are input and output
                            double & eta_t,        // --update=SAFE will MODIFY eta_t now
                            const EigenType& x, const SparseMb& y, const VectorXd& xSqNorms,
                            const double C1, const double C2, const double lambda,
                            const unsigned long t, const size_t nTrain,
                            const VectorXi& nclasses, const int maxclasses,
                            const boolmatrix& filtered,
                            MCupdateChunking const& updateSettings,
                            param_struct const& params)
        {
            // make sortlu* variables valid (if possible, and not already valid)
            luPerm.mkok_sortlu();
            luPerm.mkok_sortlu_avg();
            assert( luPerm.ok_sortlu );
            //assert( luPerm.ok_sortlu_avg ); // sortlu_avg may be undefined (until t>=epoch_avg)
            if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t    >    params.avg_epoch){
                assert( luPerm.ok_sortlu_avg );
            }
            VectorXd& sortedLU                          = luPerm.sortlu;
            VectorXd& sortedLU_avg                      = luPerm.sortlu_avg;
            std::vector<int> const& sorted_class        = luPerm.perm;
            std::vector<int> const& class_order         = luPerm.rev;

            size_t const& batch_size    = updateSettings.batch_size;
            // the following bunch should disappear soon
            int const& sc_chunks        = updateSettings.sc_chunks;
            int const& sc_chunk_size    = updateSettings.sc_chunk_size;
            int const& sc_remaining     = updateSettings.sc_remaining;

            int const& idx_chunks       = updateSettings.idx_chunks;
            int const& idx_chunk_size   = updateSettings.idx_chunk_size;
            int const& idx_remaining    = updateSettings.idx_remaining;
            MutexType* idx_locks       = updateSettings.idx_locks;
            MutexType* sc_locks        = updateSettings.sc_locks;

            // After some point 'update' BEGINS TO ACCUMULATE sortedLU into sortedLU
            assert( luPerm.ok_sortlu_avg == true ); // accumulator begins at all zeros, so true
            if (params.update_type == SAFE_SGD) {
                update_safe_SGD(w, sortedLU, sortedLU_avg, eta_t,
                                x, y, xSqNorms,
                                C1, C2, lambda, t, nTrain, // nTrain is just x.rows()
                                nclasses, maxclasses, sorted_class, class_order, filtered,
                                sc_chunks, sc_chunk_size, sc_remaining,
                                params);
            } else if (params.update_type == MINIBATCH_SGD) {
                update_minibatch_SGD(w, sortedLU, sortedLU_avg,
                                     x, y, C1, C2, lambda, t, eta_t, nTrain, batch_size,
                                     nclasses, maxclasses, sorted_class, class_order, filtered,
                                     sc_chunks, sc_chunk_size, sc_remaining,
                                     idx_chunks, idx_chunk_size, idx_remaining,
                                     idx_locks, sc_locks,
                                     params);
            }
            luPerm.chg_sortlu();        // sortlu change ==> ok_lu now false
            // After some point 'update' BEGINS TO ACCUMULATE sortedLU into sortedLU_avg
            if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch){
                assert( luPerm.ok_sortlu_avg == true );
                ++luPerm.nAccSortlu_avg;
                luPerm.chg_sortlu_avg();      // ==> {l,u}_avg are no longer OK reflections of accumulated sortlu_avg
            }
        }
};
