
#include "mcsolver.hh"

struct MCupdate
{
#if MCUC>=2 && MCPRM==0
    // version 1...
    template<typename EigenType> 
        static void update( WeightVector& w, VectorXd& sortedLU, VectorXd& sortedLU_avg,
                            const EigenType& x, const SparseMb& y,
                            const double C1, const double C2, const double lambda,
                            const unsigned long t, const double eta_t,
                            const size_t nTrain, //const size_t batch_size,
                            const VectorXi& nclasses, const int maxclasses,
                            const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                            const boolmatrix& filtered,
                            //const size_t sc_chunks, const size_t sc_chunk_size, const size_t sc_remaining,
                            //const int idx_chunks, const int idx_chunk_size, const int idx_remaining,
                            //MutexType* idx_locks, MutexType* sc_locks,
                            MCupdateChunking const& updateSettings,
                            param_struct const& params)
        {
            // make sortlu* variables valid (if possible, and not already valid)
            //luPerm.mkok_sortlu();
            //luPerm.mkok_sortlu_avg();

            //VectorXd& sortedLU                          = luPerm.sortlu;
            //VectorXd& sortedLU_avg                      = luPerm.sortlu_avg;
            //std::vector<int> const& sorted_class        = luPerm.perm;
            //std::vector<int> const& class_order         = luPerm.rev;

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

            if (params.update_type == SAFE_SGD) {
                update_safe_SGD(w, sortedLU, sortedLU_avg,
                                x, y, C1, C2, lambda, t, eta_t, nTrain, // nTrain is just x.rows()
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
            // inform that sortlu* have changed... and therefore {l,u}_avg are out of date
            //luPerm.chg_sortlu();
            //luPerm.chg_sortlu_avg();
        }
#elif MCUC>1 && MCPRM>0
    template<typename EigenType> 
        static void update( WeightVector& w, /*VectorXd& sortedLU, VectorXd& sortedLU_avg,*/
                            MCpermState & luPerm,      // sortlu and sortlu_avg are input and output
                            const EigenType& x, const SparseMb& y,
                            const double C1, const double C2, const double lambda,
                            const unsigned long t, const double eta_t,
                            const size_t nTrain, //const size_t batch_size,
                            const VectorXi& nclasses, const int maxclasses,
                            /*const std::vector<int>& sorted_class, const std::vector<int>& class_order,*/
                            const boolmatrix& filtered,
                            //const size_t sc_chunks, const size_t sc_chunk_size, const size_t sc_remaining,
                            //const int idx_chunks, const int idx_chunk_size, const int idx_remaining,
                            //MutexType* idx_locks, MutexType* sc_locks,
                            MCupdateChunking const& updateSettings,
                            param_struct const& params)
        {
            // make sortlu* variables valid (if possible, and not already valid)
            luPerm.mkok_sortlu();
            luPerm.mkok_sortlu_avg();
#if MCPRM>=1
            assert( luPerm.ok_sortlu );
            //assert( luPerm.ok_sortlu_avg ); // sortlu_avg may be undefined (until t>=epoch_avg)
            if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t    >    params.avg_epoch){
                assert( luPerm.ok_sortlu_avg );
            }
#endif
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

            if (params.update_type == SAFE_SGD) {
                update_safe_SGD(w, sortedLU, sortedLU_avg,
                                x, y, C1, C2, lambda, t, eta_t, nTrain, // nTrain is just x.rows()
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
            // inform that sortlu* have changed... and therefore {l,u}_avg are out of date
            luPerm.chg_sortlu();
            if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch){
                // Here or later, 'update' ACCUMULATES sortedLU into sortedLU
                luPerm.chg_sortlu_avg();
            }
#if MCPRM>=1
            assert( luPerm.ok_sortlu );
            if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch){
                assert( luPerm.ok_lu_avg );
            }
#endif
        }
#endif
};
