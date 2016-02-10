#ifndef MCPREDICT_H
#define MCPREDICT_H

#include "find_w.h"

/** Simplest projector returns the final bitset for each input example.
 * It calculates \c projections=x*s.weights_avg. Then it uses
 * \c s.lower_bounds_avg and \c s.upper_bounds_avg as a filter on
 * possible class labels for each example \em row of \c x.
 */
template<typename EigenType>
std::vector<boost::dynamic_bitset<>> project( EigenType const& x, MCsoln const& s );


/** Given any crude \em centrality measure (0-255) we can rank and retain some number
 * of <em>most central</em> classes, instead of just a crude \em yes/no evaluation.
 *
 * \note liblinear or other \em heavyweight classifiers are <B>intentionally not</B>
 *       part of the basic milde operation of MCfilter.
 */
struct SimpleProjectionScores
{
    std::vector<uint_least8_t> scores;  ///< crude score, by some \em simple {l,u} centrality
    std::vector<uint32_t> classes;      ///< list of classes
    uint32_t size() const {
        assert( scores.size() == classes.size() );
        return scores.size();
    }
    void add( uint32_t klass, uint_least8_t score ){    ///< call this for every class
        if( score == 0U ){
            ++nNonZero;
            return;
        }
        if( score >= minScore ){
            scores.push_back( score );
            classes.push_back( klass );
        }
    }
    void sort();        ///< sort scores and classes by descending score
    /** sort and bump up \c minScore to retain just \em above \c targetSize entries */
    void prune(uint32_t const targetSize);
    /** sort, set \c minScore to try for <= \c targetSize entries, and free memory.
     * If minScore hits 255, then may be forced to keep more than targetSize entries? */
    void finalPrune(uint32_t const targetSize);
private:
    uint32_t nNonZero;                  ///< count all adds with score != 0U
    uint_least8_t minScore;             ///< start by keeping anything > 1U
    //uint_least8_t targetSize;           ///< approx how may scores to retain
    //bool sorted;
};


/** Alternate filtering, using a centrality score.
 * \p x         row-wise input examples
 *              (for efficiency, pass as many rows at a time as you can).
 * \p s         MCsoln.  weights_avg, lower_bounds_avg and upper_bounds_avg are used,
 *              while [opt.] medians could be used to assign crude scores.
 * \p targetSize retain at most this many classes
 *
 * \return \c scores: A crude \em centrality ranking of the projection values,
 *         where \em middle ~ 255 and out-of-bounds ~ 0. 
 *
 * - if MCsoln medians exists, use a simple triangle function from low--median--high
 * - o.w. assume median at (low+high)/2 (akin to \c REORDER_RANGE_MIDPOINTS of \ref parameter.h)
 *
 * - When scores[i].nNonZero is much higher than scores[i].size(), you may be better
 *   off to use a bitset -- there is simply not much \em filtering going on for example \c i.
 */
template<typename EigenType>
std::vector<SimpleProjectionScores> project( EigenType const& x,
                                             MCsoln const& s,
                                             uint32_t const targetSize );


#endif // MCPREDICT_H
