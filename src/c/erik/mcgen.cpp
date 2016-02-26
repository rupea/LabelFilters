/** \file
 * test case generator for multi-class label filter.
 */

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include <cmath>        // acos
#include <random>
#include <fstream>
//#include "r64.h"      // lets use c++11 stuff...
#ifndef USE_LIBMCFILTER
#define USE_LIBMCFILTER 0
#endif

// fwd and function declarations
namespace mcgen {
    namespace opt {
        /** internal view of program parameters + dependent constants */
        struct Parms;

        /** translate program arguments --> \c Parms */
        void argsParse( int argc, char**argv, struct Parms& parms );

        /** program usage */
        void help( std::ostream& os );

    }//opt::
    namespace ndim {
        typedef std::vector<double> Vector;
        void randFill( Vector& v, double lo, double hi ); ///< fill v[] with rands in [lo,hi]
        template<class RandGenerator> inline
        void randCube( RandGenerator &g, Vector& v, float const lo=0.0f, float const hi=1.0f);
        template<class RandGenerator>
        void randDirn( RandGenerator &g, Vector& v);      ///< fill v[] with a rand unit vector (seq rotns alg)
        template<class RandGenerator>
        Vector&& randCube(RandGenerator &g, uint32_t const sz, float const lo=0.0f, float const hi=1.0f);
        template<class RandGenerator>
        Vector&& randDirn(RandGenerator &g, uint32_t sz); ///< resize and fill v[] with a rand unit vector (seq rotns alg)
        //randBall(Vector& v);            ///< fill v[] with vector magnitude <= 1
        /** \name vector transforms
         * Once training and solution examples are generated for a trivial set
         * of signal dimensions, it is easy to transform the training vectors into
         * other coordinate systems / higher dimensions.
         *
         * - Trivial examples begin with class separations on a few (low) axes.
         * - Rotater modifies trivial axes while retaining angles.
         *   - generally will mix signal axes into every dimension.
         *   - It is a set of Givens rotns applies to axis 0, 1, ...
         * - Skewer then changes angles (so projection axes are no longer orthogonal)
         *   - TODO check that Skewer conserves planarity.
         * - Givens is more generic, appying rotn angles to axes in
         *   any given order.
         *   - You could use this to shift signal, without Rotater,
         *     into a subset of final dimensions
         */
        //@{
        class SkewerMult;               ///< variably skew some axes towards another by multiply + rescale
        class SkewerAng;                ///< variably skew some axes towards another by rotns
        class Rotater;                  ///< Vector [random] rotation operator
        //class Translator;               ///< Vector [random] translation operator
        //class Givens;                   ///< mix one dimension into others by seq of rotns
        // TODO: matrix multiply trivial examples into higher dimensionality.
        //@}
    }//ndim::

    /** generate random examples according to opt::Parms */
    class ExampleGenerator;
}//mcgen::

// structure declarations
namespace mcgen {
    namespace opt {
        class Parms {
        public:
            Parms();                    ///< construct with defaults, and call init()

            /// \name core settings
            //@{
            uint32_t axes;              ///< [3] # projection axes to separate examples
            bool     ball;              ///< [*] true = examples on unit ball, false = rotated unit hypercube
            uint32_t dim;               ///< [3] dimension of each training example
            int32_t  margin;            ///< [+axes] |margin|<=a 0=none, +ve=separable, -ve=non-separable
            double   fmargin;           ///< [0.01] distance multiplier for margin example generation
            uint32_t parts;             ///< [2] each axis separates into parts, nClass = parts ^ axes [2^3]
            bool     trivial;           ///< [1]: soln is first \c axes unitvecs (skew,rot=0), 0: use skew,rot!=0
            uint32_t skew;              ///< [0] skew < axes dims toward (1,0,0,...) of trivial soln
            uint32_t rot;               ///< [0] then rotate rot<=axes axes randomly
            bool     noise;             ///< [false] 
            uint32_t embed;             ///< [0] !=0 => after noise, axes --> embed dims with rand normed xform
            uint32_t seed;              ///< [0] rand number seed
            uint32_t multi;             ///< [2] besides slc, generate a multi-labelling with up to this many labels per example.
            //@}
            /// \name non-core settings set (some via \c init() )
            //@{
            bool     write;             ///< write some .repo/.soln data files (text mode for now)
            uint32_t nClass;            ///< = parts ^ axes
            uint32_t nxStd;             ///< pre-determined examples 1 per nClass + (margin!=0? 2*|margin|*a*(parts-1)
            uint32_t nx;                ///< default (and silent minimum) is max(4*axes, nxStd)
            //@}

            /** Must call after after changing core variables.
             * - does not change \c margin (take care when changing \c axes).
             * \throw runtime_error for crazy settings. */
            void init();

            /** translate Parms core settings into a canonical string.
             * - minimal is "a[axes]d[dim]p[parts]"
             * - default is "a3d3p2", for 3 axes, 3 dims, 2 parts per axis;
             *   -  i.e. one separating plane per axis, for 8 separable classes. */
            std::string str();
        };
    }//opt::
    
    std::ostream& operator<<( std::ostream& os, opt::Parms const& p );

    class ExampleGenerator{
    public:
        ExampleGenerator( opt::Parms const& parms );
        /** create a new training example, \c x and its class \c y */
        void example( std::vector<double>& x, uint32_t& y );
    private:
        opt::Parms p;   ///< problem description
        uint64_t nGen;  ///< counter
    };
}//mcgen

// -------- implementation --------

#include <boost/program_options.hpp>

// declare implementation details
namespace mcgen {
    namespace opt {
        namespace po = boost::program_options;

#if 0
        // internal helper declarations
        static void init( po::options_description & desc );                     ///< initialize po
        static void helpUsage( std::ostream& os );                              ///< intro help
        //static void help( std::ostream& os, po::options_description & desc );   ///< options help
        static void helpExamples( std::ostream& os );                           ///< closing comments
#endif

    }//opt::
    namespace ndim {
        //typedef std::vector<double> Vector;
        template<class RandGenerator> inline
        void randCube( RandGenerator &g, std::vector<float>& v, float const lo/*=0.0f*/, float const hi/*=1.0f*/){
            std::uniform_real_distribution<float> equiRand( lo, hi );
            for(auto &x: v) x= equiRand(g);
        }
        template<class RandGenerator> inline
        void randDirn( RandGenerator &g, std::vector<float>& v){
            double x2=0.0;
            double x2Thresh = v.size()*1.e-12;
            do{
                std::normal_distribution<float> normal;
                x2=0.0;
                for(auto &x: v){
                    x = normal(g);
                    x2 += x * x;
                }
            }while(x2<x2Thresh);
            x2=1.0 / std::sqrt(x2);
            for(auto &x: v) x *= x2;    //normalize the random dirn
        }
        template<class RandGenerator>
        Vector&& randCube(RandGenerator &g, uint32_t const sz, float const lo/*=0.0f*/, float const hi/*=1.0f*/){
            Vector v;
            v.reserve(sz);
            std::uniform_real_distribution<float> equiRand( lo, hi );
            for(uint32_t i=0U; i<sz; ++i){
                v.push_back( equiRand(g) );
            }
            return v;
        }
        template<class RandGenerator>
        Vector&& randDirn(RandGenerator &g, uint32_t const sz){
            Vector v;
            v.reserve(sz);
            std::normal_distribution<float> normal();
            double x2=0.0;
            double x2Thresh = v.size()*1.e-12;
            for(uint32_t i=0U; i<sz; ++i){
                double x= normal(g);
                x2 += x * x;
                v.push_back(x);
            }
            while( x2<x2Thresh ){ // very rarely...
                x2 = 0.0;
                for(auto &x: v){
                    x= normal(g);
                    x2 += x * x;
                }
            }
            x2=1.0 / std::sqrt(x2);
            for(auto &x: v) x *= x2;    //normalize the random dirn
            return v;
        }
        /** SkewerMult changes the angle wrt the chosen axis, wrt
         * some origin on that axis.
         * - This can quickly generate a skewed distribution
         * - Ex. Skewer(0,-1,2.0)({x,y,z}) concerns axis 0 (i.e. x-axis)
         *   - points are shifted wrt. {-1,0,0} as the origin.
         *     - P:{x,y,z} --> { 2.0*(x-(-1)), y, z } = Q:{2x+1,y,z}
         *     - Now original |P-{-1,0,0}|^2 is L^2 = (x+1)^2+y^2+z^2
         *   - Keep L^2 constant by scaling {2x+1,y,z} by a constant.
         *     - length of Q wrt {-1,0,0} is M^2 = (2x+1)^2 + y^2 + z^2
         *     - so multiply Q (all dims) by (L^2 / M^2)
         *   - Net effect is that the vector from {-1,0,0} to P
         *     has had it's angle with the x-axis reduced.
         * \detail
         * - see also literature about von-Mies Fisher skewed normal pdf
         * - but this is simple and fast alternative
         * - Extensions
         *   - easy to work with skew along any dimn
         *   - generalizes to skew from arb. origin and arb. direction
         *     (but more memory and computation)
         *   - can be sped up if vectors support fast scaling.
         *   - could also be used to affect a subset of other dims
         *     - ex. could affect y-x angle but leave z-x angle same
         */
        class SkewerMult{
            /** Skew points P wrt projection towards/away axis \c x.
             * - Consider position \c o on \c x, and vector oP
             * - Adjust the angle between oP and axis \c x by:
             *   - Form Q by adjust dimension \c x of P by Px --> (Px -o) * \c f
             *     - i.e. Qx = f*(Px-o)
             *   - rescale Q by adjusting dimension \c x and some others to
             *     return line oQ to same length as original line oP
             * \p x skew axis (1st dimension, the x-axis, is axis x=0)
             * \p o origin on skew axis for scaling
             * \p f multiplier for scaling. >1.0 shifts points towards axis \c x
             *
             * Later:
             *
             * \p lo \p hi axes whose angle wrt x get changed [def.=all other axes]
             * - lo>=hi means change all other axes, otherwise change axes within range [lo,hi]
             */
            SkewerMult(uint32_t const x, double const o, double const f, uint32_t const lo=0, uint32_t const hi=0);
            double getF() const {return f;}
            /** Operate on any \c v with v.size() > a */
            Vector& operator()( Vector& v ) const;
        private:
            uint32_t x;         ///< "toward" axis
            double o;           ///< 
            double f;           ///< axis \c x multiplier
            uint32_t lo;        ///< start axis
            uint32_t hi;        ///< end axis
        };
        class SkewerAng{
            /** Rotates other [all|set] of axes towards axis \c a by an angle.
             * - Suppose unit vector \f$\hat{a}\f$ for axis \c a,
             *   - a vector \f$\vec{v}\f$ to point V,
             *   - and some other orthogonal axis with unit vector \f$\hat{b}\f$.
             * - coords of V are \f$v_a\f$ and \f$v_b\f$
             * - Consider offset \c o on axis \c a.
             * - +ve angle between oV and axis \c a is \f$\phi=\arccos(v_b/\sqrt(v_b^2+(v_a-o)^2))\f$
             * - reduce the angle by factor f: \f$\phi'=f\cdot \phi\f$
             * - Then skewed coords \f$v'_b\f$ and \f$v'_a\f$ of \f$\vec{v}\f$ satisfy
             *   - \f$v'_b=v_b\cos(\phi')\f$
             *   - \f$v'_a-o=(v_a-o)\sin(\theta)\f$
             * - Effect similar to SkewerMult, but slower because of explicit involvement of angles
             *
             * - Can apply to all (default) or some range of other axes in \p lo \p hi axes
             * - lo>=hi means change all other axes, otherwise change axes within range [lo,hi]
             * - \c f is adjusted downwards to 1.0 (a no-op)
             * - \c hi<=lo sets lo=hi=0 to signal default case of "all other axes"
             */
            SkewerAng(uint32_t const a, double const o, double const f, uint32_t const lo=0, uint32_t const hi=0);
            double getO() const {return o;}
            double getF() const {return f;}
            /** Operate on vector \c v */
            Vector& operator()( Vector& v ) const;
        private:
            uint32_t a;         ///< "toward" axis
            double o;           ///< 
            double f;           ///< angle contraction factor in [0,1]
            uint32_t lo;        ///< start axis
            uint32_t hi;        ///< end axis
        };
        /** Rotater rotates points v around origin by applying a sequence of Givens
         * rotations sequentially to axes.
         * - Notice that randDirn(v) could be implemented as a Rotater(v.size()-1)
         *   operating on vector (1,0,0,...). It just optimizes that calculation.
         * - This mixes "signal" axes into all other axes, in general.
         *
         * - Since expected use is for random rotn, I don't really care exactly
         *   where any particular vector ends up.
         * - o/w might be nicer to specify as the vector that x-hat goes to,
         *   - but that would mean I'd have to calculate the series of rotns.
         */
        class Rotater{
            /** n-dim random Rotater, using n-1 random Givens rotns */
            Rotater( uint32_t const n );       ///< random angles for an n-dim rotater
            Rotater( std::vector<double> const angles ); ///< using given angles
            Vector const& getAngles() const {return angles;}
            Vector& operator()( Vector& v ) const;
        private:
            Vector angles;     ///< angles.size() is v.size()-1 for random rotation
        };
#if 1+1
        /** Translater (for unit cube problems) */
        class Translater{
            Translater( uint32_t n); ///< by some amount within unit n-dim cube */
            Translater( Vector const& translation ); ///< by given vector
            Vector const& getTranslation() const {return trans;}
            Vector& operator()( Vector& v ) const;
            Vector& getTranslation();
        private:
            Vector trans;
        };
        /** A most-flexible Rotater.
         * - This transform allows one the mix signal axes into
         *   just a subset of final dimensions.
         */
        class Givens{
            /** apply angles[] sequentially to given ax[].
             * \pre angles.size() + 1 == axes.size(). */
            Givens( Vector const& angles,
                    std::vector<uint32_t> const& axes);
            Vector const& getAngles() const {return angles;}
            std::vector<uint32_t> const& axes() const {return ax;}
            /** operate on any v.size() > max element in ax[] */
            Vector& operator()( Vector& v ) const;
        private:
            Vector angles;     ///< angles.size() is ax.size()-1
            std::vector<uint32_t> const ax;
        };
#endif
    }
}//mcgen

// define implementation & details
#include <iostream>
#include <iomanip>
#include <tuple>
namespace mcgen {
    namespace ndim {
        SkewerMult::SkewerMult(uint32_t const x, double const o, double const f,
                               uint32_t const lo/*=0*/, uint32_t const hi/*=0*/)
            : x(x), o(o), f(f), lo(lo), hi(hi)
        {
            if( hi >= lo ){ const_cast<uint32_t&>(lo)=0U; const_cast<uint32_t&>(hi)=0U; }
#ifndef NDEBUG
            if( hi > lo ){
                assert( hi > lo+1U || lo != x );        // at least one element of [lo,hi) != x
            }
#endif
        }
        Vector& SkewerMult::operator()( Vector& v ) const{
            assert( v.size() < std::max(x,hi) );
            if( hi <= lo ){
                double vx = v[x];           // x-projection of v (point P)
                double vsq = -vx*vx;
                for(auto const e: v) vsq += e*e;   // sumsq of *other* dims
                vx-=o;
                double op2 = vsq + vx*vx;   // L^2 of oP, to be maintained

                vx = (v[x]-o) * f;          // scale dimension x of P
                double const oq2 = vsq + vx*vx;     // L^2 of oQ, oq2 != op2
                v[x] = vx;

                if( oq2 != 0.0 ){
                    double scal = sqrt( op2 / oq2 );
                    for(auto & e: v) e *= scal;
                }
            }else{
                double vx = v[x];           // x-projection of v (point P)
                double vsq = -vx*vx;
                for(auto const e: v) vsq += e*e;   // sumsq of *other* dims
                vx-=o;
                double op2 = vsq + vx*vx;   // L^2 of oP, to be maintained

                vx = (v[x]-o) * f;          // scale dimension x of P
                double const oq2 = vsq + vx*vx;     // L^2 of oQ, oq2 != op2
                v[x] = vx;

                if( oq2 != 0 ){
                    double ssq=0.0;
                    uint32_t nssq = 0U;
                    for(uint32_t i=lo; i<hi; ++i){
                        if( i==x )
                            continue;
                        ssq += v[i]*v[i];
                        ++nssq;
                    }
                    double const mul = sqrt( (oq2 - op2) / nssq );
                    for(uint32_t i=lo; i<hi; ++i){
                        if( i==x )
                            continue;
                        v[i] *= mul;
                    }
                }
            }
            return v;
        }
        SkewerAng::SkewerAng(uint32_t const a, double const o, double const f,
                             uint32_t const lo/*=0*/, uint32_t const hi/*=0*/)
            : a(a), o(o), f(f), lo(lo), hi(hi)
        {
            if( hi >= lo ){ const_cast<uint32_t&>(lo)=0U; const_cast<uint32_t&>(hi)=0U; }
            if( f > 1.0 ) const_cast<double&>(f) = 1.0;
#ifndef NDEBUG
            if( hi > lo ){
                assert( hi > lo+1U || lo != a );        // at least one element of [lo,hi) != x
            }
#endif
        }
        Vector& SkewerAng::operator()( Vector& v ) const{
            assert( v.size() < std::max(a,hi) );
            if( f == 1.0 )
                return v;
            uint32_t l, h;
            if( hi<=lo ) { l=lo; h=hi; }
            else         { l=0U; h=v.size(); }
            double va = v[a];
            for(uint32_t i=l; i<h; ++i){
                if( i==a )
                    continue;
                double const phi = f * std::acos(v[i] / sqrt( v[i]*v[i] + (va-o)*(va-o) ));
                v[i] *= cos(phi);
                va = (va-o)*sin(phi) + o;
            }
            return v;
        }
    }//ndim::
    namespace opt {
        // implementations
        using namespace std;
        using namespace boost;

        Parms::Parms()
            : axes( 3U )
              , ball( false )
              , dim( 3U )
              , margin( 3U )
              , fmargin( 0.01 )
              , parts( 2U )
              , trivial( true )
              , skew( 0U )
              , rot( 0U )
              , noise( false )
              , embed( 0U )
              , seed( 0U )
              , multi( 2U )
              , write( true )
              , nClass( 0U ) // set via init()
              , nxStd( 0U )     // set via init()
              , nx( 0U )        // set via init()
        {
            //init();
        }
        void Parms::init(){
            int const verbose=1;
            nClass = parts;
            if(verbose) cout<<" Parms::init";
            if(verbose>=2) {cout<<" nClass="<<parts; cout.flush();}
            for(uint32_t i=1U; i<axes; ++i){
                nClass *= parts;
                if(verbose>=2){cout<<"   "<<i<<".."<<nClass; cout.flush();}
            }
            if(verbose) cout<<" nClass=p^a="<<parts<<"^"<<axes<<"="<<nClass;
            uint32_t m = static_cast<uint32_t>((margin>0? margin: -margin));
            if( m > axes ) throw std::runtime_error("Parms::init ERROR: --margin cannot be > --axes");
            uint32_t nMargin = 0U;
            if( margin != 0 ){
                nMargin = 4U * m * (axes-1U) * (parts-1U);
                if(verbose)cout<<" nMargin=4|m|(a-1)(p-1)=4*"<<m<<"*"<<axes-1U<<"*"<<parts-1U<<"="<<nMargin;
            }
            nxStd = nClass + nMargin;
            if(verbose)cout<<" nxStd=nClass+nMargin="<<nClass<<"+"<<nMargin<<"="<<nxStd;

            uint32_t nMin = std::max(4*axes, (axes-m)*nxStd);
            nx = std::max( nxStd+nx, nMin ) - nxStd; // nx might increase, forcing some min # examples
            // - 1st of nClass nxStd examples are ~ [farthest] class centers
            // - rest of nxStd are margin-tightening examples
            // - remaining (nx-nxStd) randomly scattered over example space.
            if(verbose)cout<<" nMin="<<nMin<<" nx="<<nx<<endl;
        }

        static void helpUsage( std::ostream& os ){
            os<<
                "\n Usage:"
                "\n    mcgen [options]"
                "\nFunction: generate test problems for multiclass label filtering"
                "\n    produces training file mcgen-OPT.train.txt"
                "\n    produces suggested projection vector matrix mcgen-OPT.soln.txt"
                <<endl;
        }
        static void init( boost::program_options::options_description & desc ){
            desc.add_options()
                ("help,h", "help")

                ("axes,a", po::value<uint32_t>()->default_value(3U)
                 , "projection|soln axes, 2 <= a <= d")
                ("ball,b", po::value<bool>()->implicit_value(true)->default_value(false)
                 , " examples on unit ball")
                ("cube,c", po::value<bool>()->implicit_value(true)->default_value(true)
                 , "[*] ... or examples in [deformed? r,s] unit cube")
                ("dim,d", po::value<uint32_t>()->default_value(0U)
                 , "dimension of training data, d >= a")
                ("embed,e", po::value<uint32_t>()->default_value(0U)
                 , "embed a<=e<=d, after noise, default=a=NOOP")
                ("fmargin,f", po::value<double>()->default_value(0.01)
                 , "push factor related to margin width")
                ("margin,m", po::value<int32_t>()
                 , "[a] |m|<=a, add 2|m|a(p-1) margin tightness examples."
                 "  +ve: separable, -ve: non-separable, 0: no margin-examples")
                ("parts,p", po::value<uint32_t>()->default_value(2U)
                 , "p>=2 parts to divide each -a axis into. # classes = p^a [8]")
                ("seed", po::value<uint32_t>()->default_value(0U) , "rand seed")
                ("trivial,t"
                 , "[*] soln: unit vectors in 1st -a dimensions form projection axes")
                ("skew,s", po::value<uint32_t>()->default_value(0U)
                 , "TBD after t, rot last 0<s<a-1 dims toward 1st unit vector (1,0,0...) (axes-->non-orthog)")
                ("rot,r", po::value<uint32_t>()->default_value(0U)
                 , "TBD after t+s, rotate first r<=a dims randomly (keep lengths,angles)")
                ("noise,n", po::value<bool>()->implicit_value(true)->default_value(false)
                 , "after s,r: fill dim > axes with noise?")
                ("examples,x", po::value<uint32_t>()->default_value(0U), "extra random examples")
                ("classes,y", po::value<uint32_t>()->default_value(0U), "TBD cap number of single-y labels")
                ("multi", po::value<uint32_t>()->default_value(2U)
                 , "besides slc, gen multi-labelling up to m classes per example")
                ;
        }
        static void helpExamples( std::ostream& os ){
            os <<"Notes:"
                "\n - OPT string for filename is options in alphabetical order"
                "\n   with a, d, p (axes, dim, parts) required, and defaults omited"
                "\n   - Ex. defaults correspond to an options string of a3d3p2"
                "\n - Pre-determined examples (#)"
                "\n   - 1 example/class is a 'far center' (# = p^a = #_classes)"
                "\n   - m!=0 generates 2|m|(p-1) more per axis (# += 2|m|a(p-1))"
                "\n - Ex. default a=3 p=2 m=a"
                "\n       # = 2^3 + 2*|3|*3*(2-1) = 8 + 18 = 26 predetermined points"
                "\n       default training examples (x) = max(4*3, 2*26) = 52"
                "\n - Ex. a=3 p=2 m=0 ---> # = 8 , default (x) = max(4*3, 8) = 12"
                "\n - t trivial soln: sig in 1st a dims, higher dims ~ noise/zero"
                "\n - s skewed signal still in 1st a dims"
                "\n - r rot signal within 1st r<=a dims"
                "\n - n noise the dimensions from axes to dim (default: zero them)"
                "\n - e embed if e>a into e<=d dims"
                "\n - There may be better projections than the proposed .soln, sometimes"
                ;
        }
        /** \c p might not reflect values it would have after init() */
        std::string Parms::str(){
            Parms def;          // get default core values (well, some have been inited to nonzero)
            ostringstream s;
            s<<'a'<<axes;                               // non-optional
            if( ball != def.ball ) s<<(ball? 'b': 'c');
            if( margin != static_cast<int32_t>(axes) ) s<<'m'<<margin;
            //if( parts != def.parts ) s<<'p'<<parts;     // optional
            if( parts != 2U ) s<<'p'<<parts;
            if( ! trivial ){
                assert( skew > 0U || rot > 0U );
                if( skew != 0U ) s<<'s'<<skew;
                if( rot != 0U ) s<<'r'<<rot;
            }
            if(noise && dim > axes) s<<'n';
            if(embed>=axes) s<<'e'<<embed;
            if( dim > axes ) s<<'d'<<dim;
            if(nx>0U) s<<'x'<<nx;
            return s.str();
        }
        void argsParse( int argc, char**argv, struct Parms& parms ){
#define ARGSDEBUG 1            
#if ARGSDEBUG > 0
            cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
            for( int i=0; i<argc; ++i ) {
                cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
            }
#endif
            bool keepgoing = true;
            try {
                po::options_description desc("Options");
                init( desc );                    // create a description of the options

                po::variables_map vm;
                //po::store( po::parse_command_line(argc,argv,desc), vm );
                // Need more control, I think...
                {
                    po::parsed_options parsed
                        = po::command_line_parser( argc, argv )
                        .options( desc )
                        .positional( po::positional_options_description() ) // empty, none allowed.
                        //.allow_unregistered()
                        .run();
                    po::store( parsed, vm );
                }

                if( vm.count("help") ) {
                    helpUsage( cout );
                    cout<<desc<<endl;
                    helpExamples(cout);
                    keepgoing=false;
                }

                po::notify(vm); // at this point, raise any exceptions for 'required' args

                parms.axes = vm["axes"].as<uint32_t>();
                parms.dim = std::max( parms.axes, vm["dim"].as<uint32_t>() );
                if( vm.count("margin") ) {
                    parms.margin = vm["margin"].as<int32_t>();
                    cout<<" parms.margin --> "<<parms.margin<<endl;
                }else{
                    parms.margin = parms.axes;
                }
                parms.fmargin = vm["fmargin"].as<double>();
                parms.seed = vm["seed"].as<uint32_t>();
                parms.multi = vm["seed"].as<uint32_t>();
                parms.parts = vm["parts"].as<uint32_t>();
                parms.noise = vm["noise"].as<bool>();
                cout<<" DBG noise = "<<parms.noise<<endl;
                //if( vm.count("trivial") ) {
                //    int32_t const t = vm["trivial"].as<int32_t>();
                //    if( t != 0 && t != 1 ) throw runtime_error("-t must be zero or 1");
                //    parms.trivial = static_cast<bool>(t);
                //}
                parms.skew = vm["skew"].as<uint32_t>();
                parms.rot = vm["rot"].as<uint32_t>();
                parms.embed = vm["embed"].as<uint32_t>();
                if( parms.embed && parms.rot ){
                    cout<<" embed step makes rot unnecessary: ignoring --rot="<<parms.rot<<endl;
                    parms.rot = 0U;
                }
                //if( ! parms.trivial && (parms.skew == 0U && parms.rot == 0U))
                //    throw runtime_error("-t0 needs either -s or -r");
                if( parms.skew != 0U || parms.rot != 0U ) // supplying -s or -r implies non-trivial
                    parms.trivial = false;
                if( parms.axes < 1U )
                    throw runtime_error("-a must be > 1");
                if( parms.dim < parms.axes )
                    throw runtime_error("-a must be <= -d");
                if( parms.margin < -static_cast<int32_t>(parms.axes)
                  || parms.margin > static_cast<int32_t>(parms.axes))
                    throw runtime_error("-a must be <= -d");
                if( parms.parts > 1000U )
                    throw runtime_error("-p seems too large");
            }
            catch(po::error& e)
            {
                cerr<<"Invalid argument: "<<e.what()<<endl;
                throw;
            }
            catch(...)
            {
                cerr<<"Command-line parsing exception of unknown type!"<<endl;
                throw;
            }
            if( ! keepgoing ) exit(0);
            return;
        }
    }//opt::

    std::ostream& operator<<( std::ostream& os, opt::Parms const& p )
    {
        using namespace std;
#define WIDE(OS,N,STUFF) do \
        { \
            std::ostringstream oss; \
            oss<<STUFF; \
            os<<left<<setw(N)<<oss.str(); \
        }while(0)
        os<<"MCfilter parameters:\n";
        uint32_t const c1=23U;
        uint32_t const c2=22U;
        uint32_t const c3=23U;
        WIDE(os,c1,right<<setw(14)<<"axis "<<left<<p.axes);
        WIDE(os,c2,right<<setw(14)<<"dim "<<left<<p.dim);
        WIDE(os,c3,right<<setw(14)<<"trivial "<<left<<p.trivial);
        os<<endl;
        WIDE(os,c1,right<<setw(14)<<"parts "<<left<<p.parts);
        WIDE(os,c2,right<<setw(14)<<(p.ball? "ball ":"cube "));
        WIDE(os,c3,right<<setw(14)<<"rot "<<left<<p.rot);
        os<<endl;
        WIDE(os,c1,right<<setw(14)<<"margin "<<left<<p.margin);
        WIDE(os,c2,right<<setw(14)<<"fmargin "<<left<<p.fmargin);
        WIDE(os,c3,right<<setw(14)<<"skew "<<left<<p.skew);
        os<<endl;
        WIDE(os,c2,right<<setw(14)<<"noise "<<left<<p.noise);
        WIDE(os,c1,right<<setw(14)<<"embed "<<left<<p.embed);
        os<<endl;
        WIDE(os,c1,right<<setw(14)<<"nClass "<<left<<p.nClass);
        WIDE(os,c2,right<<setw(14)<<"nxStd "<<left<<p.nxStd);
        WIDE(os,c3,right<<setw(14)<<"nx "<<left<<p.nx);
        os<<endl;
        return os;
    }

    //namespace po=boost::program_options;
    void help( std::ostream& os ){
        boost::program_options::options_description desc("mcgen Options");
        opt::init(desc);
        opt::helpUsage(os);
        // opt::help(os,desc);
        os<<desc<<std::endl;
        opt::helpExamples(os);
    }

}//mcgen::

// -------- main program --------

#if USE_LIBMCFILTER
#include "find_w.h"
#include "printing.hh"
//#include <experimental/filesystem>      // file_size, oops gcc-5.x !
#include <sys/stat.h>                     // stat

using namespace std;
void mcSave(std::string saveBasename, MCsoln const& soln){            
    if( saveBasename.size() > 0U ){
        cout<<" In mcSave("<<saveBasename<<", MCsoln)"<<endl;
        using namespace std;
        string saveTxt(saveBasename); saveTxt.append("-txt.soln");
        cout<<" Saving to file "<<saveTxt<<endl;
        try{
            ofstream ofs(saveTxt);
            soln.write( ofs, MCsoln::TEXT, MCsoln::SHORT );
            ofs.close();
        }catch(std::exception const& e){
            cout<<"OHOH! Error during text write of demo.soln "<<e.what()<<endl;
            throw(e);
        }
        string saveBin(saveBasename); saveBin.append("-bin.soln");
        cout<<" Saving to file "<<saveBin<<endl;
        try{
            ofstream ofs(saveBin);
            soln.write( ofs, MCsoln::BINARY, MCsoln::SHORT );
            ofs.close();
        }catch(std::exception const& what){
            cout<<"OHOH! Error during binary write of demo.soln"<<endl;
            throw(what);
        }
    }
}

/** don't print the SparseMb \c bool values - we know what they look like */
template<typename DERIVED>
void dump( std::ostream& os, Eigen::SparseMatrixBase<DERIVED> const& sy ){
    cout<<"SparseMb["<<sy.rows()<<" x "<<sy.cols()<<"], size "<<sy.size()
        <<" nonZeros=data.size="<<sy.derived().data().size()<<" compressed="<<sy.derived().isCompressed()
        <<" :\n\tOuterPtrs:";
    for(uint32_t i=0U; i<sy.outerSize(); ++i){cout<<" "<<sy.derived().outerIndexPtr()[i];}
    cout<<" $\n\tInnerPtrs:";
    for(uint32_t i=0U; i<sy.derived().data().size(); ++i){
        assert( sy.derived().innerIndexPtr()[i] < sy.innerSize() );
        cout<<" "<<sy.derived().innerIndexPtr()[i];
    }
    cout<<endl;
}

#endif

using namespace std;
using namespace mcgen;

int main(int argc, char** argv)
{
    opt::Parms p;
    cout<<" default canonical args: "<<p.str()
        <<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    opt::argsParse( argc, argv, p );
    cout<<" Parameter dump:\n"<<p<<endl;
    p.init();
    cout<<" Parameter dump, after init():\n"<<p<<endl;

    string canonicalArgs = p.str();
    cout<<"\tcanonical args: "<<canonicalArgs
        <<" margin="<<p.margin<<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    p.init();
    cout<<"         canonical args: "<<canonicalArgs
        <<" margin="<<p.margin<<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    string trainFile;   // filename for training data file
    string axesFile;    // filename for 'ideal' projection axes
    {
        ostringstream os;
        os<<"mcgen-"<<canonicalArgs<<".train.txt";
        trainFile = os.str();
    }
    {
        ostringstream os;
        os<<"mcgen-"<<canonicalArgs<<".axes.txt";
        axesFile = os.str();
    }
    // XXX for now assume unit cube (easiest)
    p.ball = false;
    /** help partition a range into \c p linear parts */
    struct LinearEquipartition {
        uint32_t const n;       ///< equal parts
        float const lo;         ///< splitting lo
        float const hi;         ///< to hi
        float const range;      ///< total range (hi-lo)
        float const step;       ///< size of each part
        std::vector<float> const partn;       // beg,end,end,...,end of the partition
        LinearEquipartition(uint32_t parts) : n(parts)
                                              , lo(0.0), hi(1.0)
                                              , range(hi-lo), step(range/parts)
                                              , partn(n+1U)
        {
            auto & nc = const_cast<std::vector<float>&>(partn); // cast to non-const for construction
            for(uint32_t i=0U; i<n; ++i) nc[i] = lo + i*step;
            nc[n] = hi;
        }
        float beg(uint32_t const i) const { assert(i<n); return partn[i]; }
        float mid(uint32_t const i) const { assert(i<n); return (partn[i]+partn[i+1]) * 0.5; }
        float end(uint32_t const i) const { assert(i<n); return partn[i+1]; }
        /** fast reverse lookup of 'closest region number' */
        uint32_t lookup( float f ){
            float i = (min(max(f,lo),hi) - lo) / step;
            return min((uint32_t)i,n);  // could be too large because of rare rounding error?
        }
        float rand();
        void dump( std::ostream& os ) const {
            using namespace std;
            os<<" LinearEquipartition(parts="<<n<<")";
            for(uint32_t i=0U; i<n; ++i){
                os<<"\n\ti="<<i<<" b:m:e = "<<beg(i)<<":"<<mid(i)<<":"<<end(i);
            }
            os<<endl;
        }
    };
    /** Count in \c d dimensions, in base \c b, to enumerate spatial partitions.
     * - This is isomorphic to "counting in arbitrary base" with given number of digits.
     * - We use standard ripple-counting to generate the sequence.
     * - The count in digits[i] is \em little-endian, so \c digits[i][0] is the \em ones place
     */
    struct RippleCounter {
        uint32_t const d;       ///< dimension ~ number of digits ~ Parms::axes
        uint32_t const b;       ///< base (max value of each digit is 'base-1')
        /** digits[i] is the <b>i</b>'th <b>d</b>-dimensional vector of base-<b>b</b> digits */
        std::vector<std::vector<uint32_t>> const digits;
        RippleCounter( uint32_t dim, uint32_t base ) : d(dim), b(base), digits()
        {
            using namespace std;
            auto & dig = const_cast<std::vector<std::vector<uint32_t>>&>( digits );  // non-const for constructor
            vector<uint32_t> cnt(d,0U); // start at zero
            dig.push_back(cnt);
            //cout<<" +RippleCounter(dim="<<dim<<",base="<<base<<") digits.size() = "<<digits.size()<<endl;
            // generate 'v' in standard ripple-counting order
            uint32_t place=0U;  // who should be incrementing at the moment?
            while( place < d ){
                ++cnt[place];
                if( cnt[place] < b ){
                    dig.push_back(cnt);
                    //cout<<" digits.size() = "<<digits.size()<<endl;
                    continue;
                }// else "overflow"
                uint32_t hi=place;     // overflow into what higher digit?
                do{ ++hi; } while( hi < d && cnt[hi]+1U >= b );
                if( hi >= d ) break;   // EXIT when nowhere to overflow
                assert( hi < d );
                ++cnt[hi];
                assert( cnt[hi] < b );
                for(uint32_t lo=0U; lo<hi; ++lo) cnt[lo] = 0U;  // reset all lower digits
                dig.push_back(cnt);
                //cout<<" digits.size() = "<<digits.size()<<endl;
                place = 0U;
            }
        }
        /** for partition \c p, what are the neighboring partition indices?
         * Close spatial \em neighbors are when a single digit of \c digits[p]
         * changes to a neighboring value (no wraparound). i.e. the neighbour
         * is \em closest in the axis of the changed digit. */
        std::vector<uint32_t> neighbors( uint32_t const p )
        {
            std::vector<uint32_t> ret;
            uint32_t offset = 1U;       // how big a change of 1 in 'place' is
            for( uint32_t place=0U; place<d; offset*=b, ++place ){
                if( digits[p][place] + 1U < b ){     // digits[p+offset] is one
                    assert( p + offset < digits.size() && digits[p+offset][place] == digits[p][place] + 1U );
                    ret.push_back(p+offset);    // higher than digits[p]
                }
                if( digits[p][place] > 0U ){    // digits[p][p-offset] is one
                    assert( digits[p-offset][place] == digits[p][place] - 1U );
                    ret.push_back(p-offset);    // lower than digits[p]
                }
            }
            return ret;
        }
        std::vector<uint32_t>::size_type size() const {return digits.size();}
        /** shortcut digits[p] accesor */
        std::vector<uint32_t> const& operator[](uint32_t const p) const {
            assert( p < digits.size() );
            return digits[p];
        }
        /** digits[] --> count value, throw/assert if digits is not d-diml base-b number */
        uint32_t lookup( std::vector<uint32_t> const& dig ) const {
#ifndef NDEBUG
            assert( dig.size() == d );
            for( auto const dd: dig ) assert( dd < b );
#endif
            uint32_t ret = 0U;
            uint32_t mul = 1U;
            for(uint32_t dd=0U; dd<dig.size(); mul*=b, ++dd){
                ret += dig[dd]*mul;
            }
            return ret;
        }
        void dump( std::ostream& os ) const {
            cout<<" RippleCounter:"<<endl;
            for(uint32_t c=0U; c<digits.size(); ++c ){
                cout<<"\n\trc["<<c<<"] = {";
                for(auto const d: digits[c]) cout<<" "<<d;
                cout<<" }";
            }
            cout<<endl;
        }
    };

    vector<vector<float>> x;            // vector of examples, in lowest possible dimensionality
    vector<uint32_t> y;                 // partition of each example (initial 'class' of the example)
    // 1. generate the trivial axis solution, (a dims)
    vector<vector<float>> soln(p.axes);
    {
        for(uint32_t u=0U; u<p.axes; ++u){              // for each unit vector
            soln[u].clear(); soln[u].resize(p.axes);    //   fill w/ 0.0
            soln[u][u] = 1.0;                           //   set one axis to 1.0
        }
    }
    // 2. generate trivial examples: 1 central example per class
    RippleCounter rc(p.axes, p.parts);  // partition numbering, one digit per axis
    LinearEquipartition equi(p.parts);  // numeric range for each axis-digit
    if(1) rc.dump(cout);
    if(1) equi.dump(cout);
    vector<vector<float>> rcMid;        // for each partition of rc, store the coords
    {
        vector<float> central(p.axes);
        for(uint32_t c=0U; c<rc.digits.size(); ++c){     // for each class 'c'
            auto digits = rc[c];                        //   get 'digits' of this class (spatial partition)
            for(uint32_t a=0U; a<digits.size(); ++a)    //   for each axis...
                central[a] = equi.mid( digits[a] );     //      push it's midpoint value
            rcMid.push_back( central );                 //   save axes-dimensional central value
        }
    }
    for(uint32_t c=0U; c<rcMid.size(); ++c){
        x.push_back( rcMid[c] );
        y.push_back( c );
    }
    uint32_t nxMidpoint = x.size();
    if(1){ // verify rc.lookup function
        for(uint32_t r=0U; r<rc.size(); ++r){
            auto digits = rc[r];
            uint32_t lookup = rc.lookup(digits);
            assert( lookup == r );
        }
    }
    if(1){
        cout<<" class central point examples: x.size()="<<x.size()<<" nxStd="<<p.nxStd;
        for(uint32_t i=0U; i<nxMidpoint; ++i){
            cout<<"\n\t x["<<i<<"] = {";
            for( auto const xxx: x[i] )
                cout<<" "<<xxx;
            cout<<" }";
        }
        cout<<endl;
    }
    // 3a. generate all nearest-neighbor midpoints (for all axes)
    /** Nearest-neighbor partitions and midpoints.
     * - \c aa stores a lower neighbor, \c bb stores a \em higher neighbor
     * - \c mid stores vector of coordinates.
     *
     * - partition correspond to a partition number < \c rc.b=Parm::parts on each axis
     *   - partition corresponds to 'digits' of a base \c rc.b number
     *   - nearest-neighbor \c bb of \c aa has identical digits, except for
     *     one axis whose digit differs by one
     *   - store list of neighbors ∋  <tt>bb</tt>'s digit is \em higher,
     *     to avoid counting pairs twice
     *   - digits of each partition are stored in RippleCounter [0,1,...]
     *   - Each digit corresponds to coordinate value \c equi.mid(digit) in a \c LinearEquipartition
     * - \b hardcoded to construct midpoints with \c rcMid[]
     */
    struct Mid {
        static std::vector<float> midpoint( std::vector<float> const& a, std::vector<float> const& b ){
            std::vector<float> ret;
            ret.reserve( a.size() );
            for(uint32_t i=0U; i<a.size(); ++i){
                ret.emplace_back( (a[i] + b[i]) * 0.5 );
            }
            return ret;
        }
        uint32_t const aa;              ///< partition, coords rcMid[aa]
        uint32_t const bb;              ///< partition, rc[bb] has a digit one higher than rc[aa]
        uint32_t const ax;              ///< which axis differs, so rc[bb][ax] == rc[aa][ax] + 1
        std::vector<float> const mid;   ///< midpoint coords
        Mid(std::vector<std::vector<float>> const& rcMid
            , uint32_t const aa, uint32_t const bb, uint32_t const ax)
            : aa(aa), bb(bb), ax(ax), mid( midpoint( rcMid[aa], rcMid[bb] ))
        {}
    };
    vector<Mid> mid;            ///< with \em all the nearest-neighbor midpoints
    {// RippleCounter rc + rcMid ---> vector<Mid> mid
        int const verbose = 0;
        // rc.neighbors is instructional -- we want to do that,
        //vector<tuple<uint32_t,uint32_t>> pairs; // margins are between pairs of neighbors
        for( uint32_t aa=0U; aa<rc.size(); ++aa){                  // for each start partition
            if(verbose) {cout<<" aa="<<aa; cout.flush(); }
            uint32_t offset = 1U; // how big a change of 1 in 'place' is
            for( uint32_t place=0U; place<rc.d; offset*=rc.b, ++place ){
                //                        ^^^^ for every axis
                if(verbose){ cout<<"   place,off="<<place<<","<<offset; cout.flush();}
                // can we bump that digit up by one?
                uint32_t neighbor_hi = aa + offset;
                if( rc[aa][place] + 1U < rc.b ){ // digits[p+offset] is one higher
                    if(verbose){cout<<" YES"; cout.flush();}
                    assert( rc.digits[neighbor_hi][place] == rc.digits[aa][place] + 1U );
                    mid.emplace_back(Mid(rcMid,aa,neighbor_hi,place));
                }else if(verbose){cout<<" no "; cout.flush();}
                // don't consider digit --> digit - 1 -- this would have all margins twice
            }
            if(verbose){cout<<endl;}
        }
        if(1){
            cout<<" ALL margin-neighbors connect partition numbers (a,b)axis :"<<endl;
            for(auto const& m: mid) cout<<" ("<<m.aa<<","<<m.bb<<")a"<<m.ax;
            cout<<endl<<"\t";
            for(uint32_t i=0U; i<std::min((uint32_t)10U, (uint32_t)mid.size()); ++i){
                cout<<"#"<<left<<setw(3)<<i<<" ("<<mid[i].aa<<","<<mid[i].bb<<")a"<<mid[i].ax<<"\n\t\t  a@{";
                for(auto x: rcMid[mid[i].aa]) cout<<" "<<x; cout<<" }\n\t\t  b@{";
                for(auto x: rcMid[mid[i].bb]) cout<<" "<<x; cout<<" }\n\t\tmid@{";
                for(auto x: mid[i].mid) cout<<" "<<x; cout<<" }";
                cout<<endl<<"\t";
            }
            for(auto const&p: mid){             // print all original digits
                auto const& digits = rc[p.aa];
                cout<<"{"; for(auto const d: digits) cout<<" "<<d; cout<<"}";
            }
            cout<<endl<<"\t";
            for(auto const&p: mid){             // print higher-neighbor digits
                auto const& digits = rc[p.bb];  // only 1 digit has been bumped up
                cout<<"{"; for(auto const d: digits) cout<<" "<<d; cout<<"}";
            }
            cout<<endl<<endl;
        }
    }
    // push point a by factor f towards point b
    auto pushpoint = []( std::vector<float>& a, double const f, std::vector<float> const& b ){
        assert( a.size() == b.size() );
        if( f != 0.0 ){
            double const g = 1.0-f;
            for(uint32_t i=0U; i<b.size(); ++i){
                a[i] = g * a[i] + f * b[i];
            }
        }
    };
    // find nearest rc partition to random point r
    auto nn = [&rc, &equi]( std::vector<float> const& r ) -> uint32_t {
        assert( r.size() == rc.d );
        vector<uint32_t> dr;    // What are per-axis closes 'digits' of r?
        dr.reserve(rc.d);
        for(uint32_t a=0U; a<r.size(); ++a){   // for each axis
            dr.push_back( equi.lookup( r[a] )); //   lookup equipartition 'digit'
        }
        //cout<<" nn";for(auto x:dr)cout<<","<<x; cout<<" "; cout.flush();
        return rc.lookup(dr);   // lookup those digits in the RippleCounter, rc.
    };
    if(1){ // verify margin operations --- correct [fast] partition lookup
        // RESULT: If exactly on margin, digits 'lookup' is not unique
        //         so we get SOME close neighbor but not nec. the one we started from.
        // test quick lookup of every margin point to its partition
        // using a 'lookup' on each axis separately to generate the 'digits'
        for(uint32_t i=0U; i<mid.size(); ++i){
            bool verbose=false;
            if(verbose){cout<<" partition i="<<i<<endl;}
            for(uint32_t n=0U; n<2U; ++n){
                uint32_t nbr = (n==0U? mid[i].aa: mid[i].bb); // which neighbor to test
                auto pm = mid[i].mid;                           // pm = Point on Margin
                pushpoint( pm, 1.e-1, rcMid[nbr] );             // push a wee bit toward aa;
                if(verbose){
                    cout<<"   nbr "<<setw(3)<<nbr<<" {"; for(auto x: rcMid[nbr]) cout<<" "<<x; cout<<" }"<<endl;
                    cout<<"    pm     {"; for(auto x: pm) cout<<" "<<x; cout<<" }"<<endl;
                    cout<<"   pushed  {"; for(auto x: pm) cout<<" "<<x; cout<<" }"<<endl;
                }
                vector<uint32_t> dpm;           // vector of Digits of Point on Margin
                for(uint32_t a=0U; a<pm.size(); ++a){
                    if(verbose){cout<<" equi.lookup( "<<pm[a]<<" ) = "<<equi.lookup(pm[a])<<endl; cout.flush();}
                    dpm.push_back( equi.lookup( pm[a] ));       // coord --> partn 'digit'
                }
                if(verbose){
                    cout<<endl;
                    cout<<" nbr {"; for(auto x: rc[nbr]) cout<<" "<<x; cout<<" }"<<endl;
                    cout<<" dpm {"; for(auto x: dpm) cout<<" "<<x; cout<<" }";
                }
                uint32_t err = 0U;
                for(uint32_t d=0U; d<dpm.size(); ++d){
                    if( dpm[d] != rc[nbr][d] ) { ++err; }
                    assert( dpm[d] == rc[nbr][d] );       // assert dpm digits correct.
                }
                if(verbose){cout<<" err="<<err<<endl;}
                assert( err == 0U );
                // test the 'nn' lambda function.  'pm' was pushed toward 'nbr', so...
                assert( nn( pm ) == nbr );
            }
        }
        cout<<" (quick lookup of nearest partition && nn lambda are OK)"<<endl;
    }
    auto dot = [] ( vector<float> const& a, vector<float> const& b ) -> float {
        float ret=0.0;
        for(uint32_t i=0U; i<a.size(); ++i) ret += a[i] * b[i];
        return ret;
    };
    if(1){ // sample code to find {l,u} bounds for fully separable case, and print
        // to find {l,u} bounds of each soln, using FULL set of margin points, with strict +ve margin
        vector<vector<float>> l (soln.size(), std::vector<float>(rc.size(),numeric_limits<float>::max()));
        vector<vector<float>> u (soln.size(), std::vector<float>(rc.size(),numeric_limits<float>::min()));
        { // verify that soln shatters trivial x,y examples
            float const f = fabs(p.fmargin);            // use STRICTLY POSITIVE margin
            for(auto const& m: mid){                    // for all neighbor pairs
                auto v = m.mid;
                for(uint32_t n=0U; n<2U; ++n){
                    uint32_t nbr = (n==0U? m.aa: m.bb); // for bisectors shifted a bit each way
                    pushpoint( v, f, rcMid[nbr] );
                    for(uint32_t s=0U; s<soln.size(); ++s){     // for each soln unit vector
                        float vdots = dot( v, soln[s] );
                        l[s][nbr] = min( l[s][nbr], vdots );    // update l
                        u[s][nbr] = max( u[s][nbr], vdots );    // and u bounds
                    }
                }
            }
            if(1){ //print, you can very that every l,u pairs is shattered when all solns considered
                for(uint32_t s=0U; s<soln.size(); ++s){
                    cout<<"\tl["<<s<<"] = {"; for(auto ss:l[s]) cout<<" "<<setw(10)<<ss; cout<<" }\n";
                    cout<<"\tu["<<s<<"] = {"; for(auto ss:u[s]) cout<<" "<<setw(10)<<ss; cout<<" }\n";
                }
            }
        }
    }
    // 3. generate trivial solutions: according the p.margin
    {// generate some examples from margin points
        // margin = |p.margin| --> accept neighbors differing along axes 0..margin-1
        uint32_t margin = (p.margin>0? p.margin: -p.margin);   // absolute value-->unsigned
        assert( margin <= p.axes );
        // rc.neighbors is instructional -- we want to do that,
        // - but only for p.margin axes
        // - and we will grab non-repeating pairs of neighbors
        vector<tuple<uint32_t,uint32_t>> pairs; // margins are between pairs of neighbors
        float f = p.fmargin;
        if( p.margin < 0 ) f = -f;
        for(auto const& m: mid){
            if( m.ax < margin ){ // if |p.margin| < p.axes, use a subset of all margin points
                // TODO margin shift from m.mid towards/away m.aa and m.bb
                // one point exactly on margin in class aa
                x.push_back( m.mid ); pushpoint( x.back(), f, rcMid[m.aa] );
                y.push_back( m.aa  );
                cout<<" y="<<m.aa<<" x["<<x.size()<<"].back={";for(auto xx:x.back())cout<<" "<<xx;cout<<" }"<<endl;
                // other point exactly on margin in class bb
                x.push_back( m.mid ); pushpoint( x.back(), f, rcMid[m.bb] );
                y.push_back( m.bb  );
                cout<<" y="<<m.bb<<" x["<<x.size()<<"].back={";for(auto xx:x.back())cout<<" "<<xx;cout<<" }"<<endl;
            }
        }
    }
    if( x.size() != p.nxStd ){
        cout<<"ERROR:  x.size() == "<<x.size()<<", but expected p.nxStd="<<p.nxStd<<endl;
        exit(0);
    }
    uint32_t const nxMargin = x.size();
    if(1){
        cout<<" Margin points:";
        for(uint32_t i=nxMidpoint; i<nxMargin; ++i){
            cout<<"\n\tx["<<setw(3)<<i<<"] class "<<setw(3)<<y[i]<<" @{";
            for(auto xi: x[i]) cout<<" "<<xi;
            cout<<"}";
        }
        cout<<endl;
    }
    // 4. generate random examples (retry if any fail +ve p.margin)
    std::mt19937_64 gen( uint64_t{0x12345678U}  );
    double const fshrink = (p.margin<0? -p.fmargin: +p.fmargin);
    if(p.nx>0U){
        // Test rand pt R for "in margin" is not easy/speedy.
        // Instead move R so that it 'satisfies' margin settings:
        // - Original thought:
        //   - find closest partn a (by looking at 'digits' of each axes independently) FAST
        //   - find closest neighbor b
        //   - project onto line rcMid[a]--rcMid[b],
        //   - scale that projection by (+/-)fmargin
        //   - and move R correspondingly
        // - OR, EVEN SIMPLER
        //   - use 'nn' to get closest partition midpoint P
        //   - shrink R by f' towards P (ball-like shrinkage, maybe good-enough)
        //   - f' == f.margin guarantees same shrinkage as margin exemplars
        //
        vector<float> r(p.axes); // work vector
        assert( rc.d == p.axes );
        // shrink "just like" p.margin && p.fmargin would do
        //   (i.e. same-width margin plane)
        //   \return original partn assignment of r
        auto nnShrink = [&rcMid,&fshrink,&nn,&pushpoint]( std::vector<float> & r ) -> uint32_t {
            uint32_t ret = nn(r);
            vector<float> const& to = rcMid[ ret ];       // 'to' midpoint of nn partn
            pushpoint( r, fshrink, to );                  // move 'r' toward 'to' (if fshrink>0)
            return ret;
        };
        std::uniform_real_distribution<float> equiRand( equi.lo, equi.hi );
        for(uint32_t i=nxMargin; i<nxMargin+p.nx; ++i){
            for(uint32_t i=0U; i<r.size(); ++i){
                r[i] = equiRand(gen);
            }
            uint32_t const partn = nnShrink( r );       // margin-adjust r
            x.push_back( r );
            y.push_back( partn );
        }
    }
    if(1){
        cout<<" "<<p.nx<<" random, margin-respecting examples: x.size()="<<x.size()<<"\n";
        for(uint32_t i=nxMargin; i<x.size(); ++i){
            cout<<"\tx["<<setw(3)<<i<<"] y="<<setw(3)<<y[i]<<" @ {";
            for(auto const xi: x[i]) cout<<" "<<xi;
            cout<<" }\n";
        }
    }
    if( x.size() != p.nxStd + p.nx ){
        cout<<"ERROR:  x.size() == "<<x.size()<<", but expected p.nxStd+p.nx="<<p.nxStd<<"+"<<p.nx<<"="<<p.nxStd+p.nx<<endl;
        exit(0);
    }
    cout<<" fshrink = "<<fshrink<<endl;
    // 4b. milde 'slc' repos like examples sorted by class.
    vector<uint32_t> perm(y.size());
    std::iota( perm.begin(), perm.end(), 0U);
    std::stable_sort( perm.begin(), perm.end(), [&y](uint32_t const a, uint32_t const b){return y[a]<y[b];} );
    if(1){
        cout<<" all examples, class-sorted:"<<endl;
        for(uint32_t i=0; i<x.size(); ++i){
            cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
            for(auto const xi: x[perm[i]]) cout<<" "<<xi;
            cout<<" }\n";
        }
    }

    // 5. generate label mappings by mapping partn #s y[i] --> multilabel sets
    std::vector<std::vector<uint32_t>> ymap(rc.size());
    // ymap[ y[i] ] is now a VECTOR of labels
    {
        // use a FIXED multi=2 relabelling
        // - each of the nClassrc.size() original partitions is remapped to TWO
        //   class labels
        // - p --> {p, (p+1)%nClass}
        uint32_t const multi = 2U;
        if( p.multi != 2U ){
            cout<<" WARNING: --multi settings other than 2 are NOT IMPLEMENTED"<<endl;
        }
        uint32_t nClass = rc.size();
        for(uint32_t i=0U; i<nClass; ++i){      // for each class label
            ymap[i].resize(multi);
            for(uint32_t j=0U; j<multi; ++j){   // create a vector of labels
                ymap[i][j] = (i+j)%nClass;
            }
        }
        // NOTE: probably I can fully split for any multi up to p.axes,
        // Anyhow, afer the filtering, there should now by
        // only multi==2 remaining classes.
    }
    // 6a. generate any skew transforms of trivial soln
    // TODO
    // 6b. generate any rot transforms of trivial soln
    // TODO
    // 7a. expand trivial soln into higher dim with noise/zeros
    if(p.dim > p.axes){
        // soln expands by adding zeros
        for(uint32_t i=0U; i<soln.size(); ++i){
            soln[i].resize(p.dim);
        }
        std::uniform_real_distribution<float> equiRand( equi.lo, equi.hi );
        cout<<" p.dim="<<p.dim<<" > p.dim="<<p.dim<<", p.noise = "<<p.noise<<endl;
        for(uint32_t i=0U; i<x.size(); ++i){
            auto & xi = x[i];
            xi.resize( p.dim );
            if(p.noise){        // fill higher dims with rand unif noise?
                for(uint32_t j=p.axes; j<p.dim; ++j){
                    xi[j] = equiRand(gen) ;
                }
            }
        }
        // Q: also expand the midpoint helper info?
        // A: No, this could mean recreating (to reflect to noise midpoints in x!!)
        cout<<"exanded to "<<p.dim<<", x.size()="<<x.size()<<(p.noise?" with noise":"")<<endl;
        if(1){
            cout<<" all examples, class-sorted, expanded to dim="<<p.dim<<" :"<<endl;
            for(uint32_t i=0; i<x.size(); ++i){
                cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
                for(auto const xi: x[perm[i]]) cout<<" "<<xi;
                cout<<" }\n";
            }
        }
    }
    // 7b. rand full-rank rotn of signal unit vectors to mix signal into noise dims
    cout<<" p.embed = "<<p.embed<<endl;
    if(p.embed){ //p.embed==0U **skips** this transform
        // axis basis vector transformation defined by unit-vector remappings
        vector<vector<float>> abase(p.axes, vector<float>(p.dim,0.0f));
        assert( p.embed <= p.dim );
        assert( p.embed >= p.axes );
        assert( abase[0].size() == p.dim );
        // NOTE: basic need is random [ axes x embed ] submatrix of abase[axes x dim]
        if( p.embed >= p.dim ){ 
            for(uint32_t u=0U; u<p.axes; ++u){ // for each unit-vector, replace with random dirn
                double dotmax=0.0;
                do {
                    ndim::randDirn( gen, abase[u] );
                    dotmax = 0.0;
                    for(uint32_t j=0U; j<u; ++j){ 
                        double dot=0.0;
                        for(uint32_t i=0U; i<abase[u].size(); ++i)   // dot(abase[u],abase[j])
                            dot += abase[u][i] * abase[j][i];
                        dotmax = std::max( dot, std::fabs(dot) );
                    }
                }while( dotmax > 0.999 );  // try again if 2 unit vectors xform to nearly same direction
            }
        }else{ // embed into subset of p.dim dimensions
            vector<float> emb( p.embed, 0.0f );                                  // 1st embed into 'emb'
            assert( emb.size() < abase[0].size() );
            cout<<" embed="<<emb.size()<<" p.embed="<<p.embed<<endl;
            for(uint32_t u=0U; u<p.axes; ++u){ // each axis unit vector --> 'emb' in higher dim
                assert( abase[u].size() == p.dim );
                double dotmax=0.0;
                do {
                    ndim::randDirn( gen, emb );                 // generate rand 'emb'
                    dotmax = 0.0;
                    //cout<<" u="<<u; cout.flush();
                    for(uint32_t j=0U; j<u; ++j){ 
                        double dot=0.0;
                        for(uint32_t i=0U; i<emb.size(); ++i)   // dot(emb,abase[u])
                            dot += emb[i] * abase[j][i];
                        //cout<<" dot="<<dot; cout.flush();
                        dotmax = std::max( dot, std::fabs(dot) );
                    }
                    //cout<<" dotmax="<<dotmax; cout.flush();
                }while( dotmax > 0.999 );                       // until 'emb' in new direction
                for(uint32_t i=0U; i<p.embed; ++i) abase[u][i] = emb[i];        // 2nd emb --> abase[u]
                //for(uint32_t i=emb.size(); i<p.dim; ++i) abase[u][i] = 0.0f;
                //cout<<" done"<<endl;
            }
        }
        if(1){
            cout<<"axes unit-vecs --> new basis abase["<<abase.size()<<"]:"<<endl;
            for(uint32_t a=0U; a<abase.size(); ++a){
                cout<<"\tabase["<<a<<"] = {";
                for(auto const& aa: abase[a]) cout<<" "<<aa;
                cout<<" }, dim "<<abase[a].size()<<endl;
            }
        }
        // apply abase basis transform to both examples and solns
        {
#if 0
            for(uint32_t d=0U; d<abase.size(); ++d){
                for(auto const& xx: x){
                    vector<float> const& dnew = abase[d]; // new unit vector for dimension d
                    cout<<"xx.size = "<<xx.size()<<" dnew.size = "<<dnew.size()<<endl;
                }
                for(auto & s: soln){                   // same basis xform for soln vectors
                    cout<<" soln.size = "<<s.size()<<endl;
                }
            }
#endif
            for(uint32_t d=0U; d<abase.size(); ++d){
                vector<float> const& dnew = abase[d]; // new unit vector for dimension d
                for(auto & xx: x){
                    assert( xx.size() == dnew.size() );
                    double const xxd = xx[d];                   // old 'd' component.
                    xx[d] = 0.0;
                    for(uint32_t i=0U; i<dnew.size(); ++i)
                        xx[i] += xxd * dnew[i];                 // is now in dnew dirn
                }
                for(auto & xx: soln){                   // same basis xform for soln vectors
                    assert( xx.size() == dnew.size() );
                    double const xxd = xx[d];                   // old 'd' component.
                    xx[d] = 0.0;
                    for(uint32_t i=0U; i<dnew.size(); ++i)
                        xx[i] += xxd * dnew[i];                 // is now in dnew dirn
                }
            }
        }
        if(1){
            cout<<" all examples, class-sorted, embedded into embed="<<p.embed<<" :"<<endl;
            for(uint32_t i=0; i<x.size(); ++i){
                cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
                for(auto const xi: x[perm[i]]) cout<<" "<<xi;
                cout<<" }\n";
            }
            cout<<" all solns, class-sorted, embedded into embed="<<p.embed<<" :"<<endl;
            for(uint32_t i=0; i<soln.size(); ++i){
                cout<<"\tsoln["<<setw(3)<<i<<"] = {"; for(auto const si: soln[i]) cout<<" "<<si; cout<<" }\n";
            }
        }
    }


    // 9a. generate training files x,y : mcgen-slc-dr4.repo ("slc","dr4") in text format
    if(1){
        vector<uint32_t> perm(y.size());
        std::iota( perm.begin(), perm.end(), 0U);
        std::stable_sort( perm.begin(), perm.end(), [&y](uint32_t const a, uint32_t const b){return y[a]<y[b];} );
        if(1){
            cout<<" all examples, class-sorted:"<<endl;
            for(uint32_t i=0; i<x.size(); ++i){
                cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
                for(auto const xi: x[perm[i]]) cout<<" "<<xi;
                cout<<" }\n";
            }
        }
        string fname;
        {
            stringstream oss;
            oss<<"mcgen-"<<p.str()<<"-slc-dr4.repo";
            fname = oss.str();
        }
        ofstream ofs(fname);
        string canonicalArgs = p.str();
        ofs<<"## mcgen trivial cube data -- canonical "<<canonicalArgs
            <<" margin="<<p.margin<<" fmargin="<<p.fmargin<<endl;
        ofs<<"# "<<x.size()<<" "<<p.axes<<"\n"; // # <training examples> <dimensionality>
        // now output the slc labels
        for(uint32_t i=0U; i<y.size(); ++i){
            ofs //<<"L"
                <<y[perm[i]];
            // *********** IMMEDIATELY FOLLOWED ************ by the training data (grr, milde-repo.pdf)
            ofs<<" ";
            auto const& xp = x[perm[i]];
            assert( xp.size() == p.dim );
            for(uint32_t a=0U; ; ){
                ofs <<setw(8)
                    <<xp[a];
                if( ++a >= xp.size() )
                    break;
                ofs<<" ";
            }
            ofs<<"\n";
        }
        ofs.close();
        cout<<" Generated "<<fname<<endl;       // mcgen-slc-dr4.repo
    }
    // 9b. generate training files x,y : mcgen-PARMS-mlc-dr4.repo ("slc","dr4") in text format
    if(1){
        vector<uint32_t> perm(y.size());
        std::iota( perm.begin(), perm.end(), 0U);
        std::stable_sort( perm.begin(), perm.end(), [&y](uint32_t const a, uint32_t const b){return y[a]<y[b];} );
        string fname;
        {
            stringstream oss;
            oss<<"mcgen-"<<p.str()<<"-mlc-dr4.repo";
            fname = oss.str();
        }
        ofstream ofs(fname);
        string canonicalArgs = p.str();
        ofs<<"## mcgen trivial cube data -- canonical "<<canonicalArgs
            <<" margin="<<p.margin<<" fmargin="<<p.fmargin<<endl;
        ofs<<"# "<<x.size()<<" "<<p.axes<<"\n"; // # <training examples> <dimensionality>
        // now output the slc labels
        for(uint32_t i=0U; i<y.size(); ++i){
            auto ylabels = ymap[ y[perm[i]] ];
            assert( ylabels.size() >= 1U );
            ofs<<ylabels.size()<<" ";
            for(uint32_t l=0U; ; ){
                ofs //<<"L"
                    <<ylabels[l];
                if( ++l >= ylabels.size())
                    break;
                ofs<<" ";
            }
            // oh. let's keep going with the 'x' data.  *** milde_repo.pdf *** is S-O-O-O-O unclear.
            ofs<<" ";
            // now output dense real4 training vectors, p.axes floats ON REST OF SAME LINE
            auto const& xp = x[perm[i]];
            assert( xp.size() == p.dim );
            for(uint32_t a=0U; ; ){
                ofs <<setw(8)
                    <<xp[a];
                if( ++a >= xp.size() )
                    break;
                ofs<<" ";
            }
            ofs<<"\n";
        }
        ofs.close();
        cout<<" Generated "<<fname<<endl;       // mcgen-mlc-dr4.repo
    }
    // 9e. generate training file : mcgen-PARMS-slc-sr4.repo
    if(1){
        vector<uint32_t> perm(y.size());
        std::iota( perm.begin(), perm.end(), 0U);
        std::stable_sort( perm.begin(), perm.end(), [&y](uint32_t const a, uint32_t const b){return y[a]<y[b];} );
        if(0){
            cout<<" all examples, class-sorted:"<<endl;
            for(uint32_t i=0; i<x.size(); ++i){
                cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
                for(auto const xi: x[perm[i]]) cout<<" "<<xi;
                cout<<" }\n";
            }
        }
        string fname;
        {
            stringstream oss;
            oss<<"mcgen-"<<p.str()<<"-slc-sr4.repo";
            fname = oss.str();
        }
        ofstream ofs(fname);
        string canonicalArgs = p.str();
        ofs<<"## mcgen trivial cube data -- canonical "<<canonicalArgs
            <<" margin="<<p.margin<<" fmargin="<<p.fmargin<<endl;
        ofs<<"# "<<x.size()<<" "<<p.axes<<"\n"; // # <training examples> <dimensionality>
        // now output the slc labels
        for(uint32_t i=0U; i<y.size(); ++i){
            ofs //<<"L"
                <<y[perm[i]];
            // *********** IMMEDIATELY FOLLOWED ************ by the training data (grr, milde-repo.pdf)
            ofs<<" ";
            auto const& xp = x[perm[i]];
            assert( xp.size() == p.dim );       // xp always dense here
            {
                uint32_t nnz=0U;
                for(auto const xxp: xp) if( xxp != 0 ) ++nnz;
                ofs <<setw(5)<<right<<nnz;
            }
            for(uint32_t a=0U, col=0U; ; ){
                if( xp[a] != 0.0 ){
                    // Note: milde repo load does NOT use ":" as the sparse separator
                    ofs <<" "<<setw(5)<<right<<col<<" "<<setw(8)<<left<<xp[a];
                }
                if( ++a >= xp.size() )
                    break;
                ++col;
            }
            ofs<<"\n";
        }
        ofs.close();
        cout<<" Generated "<<fname<<endl;       // mcgen-slc-dr4.repo
    }
    // 9f. generate training file : mcgen-PARMS-mlc-sr4.repo
    {
        vector<uint32_t> perm(y.size());
        std::iota( perm.begin(), perm.end(), 0U);
        std::stable_sort( perm.begin(), perm.end(), [&y](uint32_t const a, uint32_t const b){return y[a]<y[b];} );
        if(0){
            cout<<" all examples, class-sorted:"<<endl;
            for(uint32_t i=0; i<x.size(); ++i){
                cout<<"\tx["<<setw(3)<<perm[i]<<"] y="<<setw(3)<<y[perm[i]]<<" @ {";
                for(auto const xi: x[perm[i]]) cout<<" "<<xi;
                cout<<" }\n";
            }
        }
        string fname;
        {
            stringstream oss;
            oss<<"mcgen-"<<p.str()<<"-mlc-sr4.repo";
            fname = oss.str();
        }
        ofstream ofs(fname);
        string canonicalArgs = p.str();
        ofs<<"## mcgen trivial cube data -- canonical "<<canonicalArgs
            <<" margin="<<p.margin<<" fmargin="<<p.fmargin<<endl;
        ofs<<"# "<<x.size()<<" "<<p.axes<<"\n"; // # <training examples> <dimensionality>
        // now output the slc labels
        for(uint32_t i=0U; i<y.size(); ++i){
            auto ylabels = ymap[ y[perm[i]] ];
            assert( ylabels.size() >= 1U );
            ofs<<ylabels.size()<<" ";
            for(uint32_t l=0U; ; ){
                ofs //<<"L"
                    <<ylabels[l];
                if( ++l >= ylabels.size())
                    break;
                ofs<<" ";
            }
            // *********** IMMEDIATELY FOLLOWED ************ by the training data (grr, milde-repo.pdf)
            ofs<<" ";
            auto const& xp = x[perm[i]];
            assert( xp.size() == p.dim );       // xp always dense here
            {
                uint32_t nnz=0U;
                for(auto const xxp: xp) if( xxp != 0 ) ++nnz;
                ofs <<setw(5)<<right<<nnz;
            }
            for(uint32_t a=0U, col=0U; ; ){
                if( xp[a] != 0.0 ){
                    // Note: milde repo load does NOT use ":" as the sparse separator
                    ofs <<" "<<setw(5)<<right<<col<<" "<<setw(8)<<left<<xp[a];
                }
                if( ++a >= xp.size() )
                    break;
                ++col;
            }
            ofs<<"\n";
        }
        ofs.close();
        cout<<" Generated "<<fname<<endl;       // mcgen-slc-dr4.repo
    }
    // 9c. generate test set (all rand) : mcgen-slc-dr4.test
    // 9d. generate test set (all rand) : mcgen-mlc-dr4.test
#if USE_LIBMCFILTER
    // 10a. create disk files that could be be used for milde "external" disk_array (with some work)
    //      These are just binary "eigen" dumps.
    if(1){
        using namespace detail; // eigen_io_bin
        string fname;
        { ostringstream oss; oss<<"mcgen-"<<p.str()<<"-x-D.bin"; fname = oss.str(); }
        { // write x into fname, dense binary
            //DenseM  ex;       // 96 elements in 784 bytes
            DenseMf ex;         // 96 elements in 400 bytes
            {
                // x is plain vector<vector<float>>     --> Eigen DenseMf
                assert( x.size() == p.nxStd + p.nx );
                assert( x[0].size() == p.dim );
                ex.resize( x.size(), p.dim );
                for(uint32_t r=0U; r<x.size(); ++r){
                    for(uint32_t c=0U; c<p.dim; ++c ){
                        ex.coeffRef(r,c) = static_cast<float>(x[r][c]);
                    }
                }
                //cout<<"x as SparseMf:\n"<<ex<<endl;
                ofstream ofs(fname);
                eigen_io_bin( ofs, ex ); // x is Dense
                ofs.close();
            }
            uint64_t fsize_bytes;
            {
                struct stat st;
                stat(fname.c_str(), &st);
                fsize_bytes = st.st_size;         // total size, in bytes
            }
            cout<<" Wrote 'x' file "<<fname<<" [ "<<ex.rows()<<" x "<<ex.cols()<<" ] : "
                <<ex.rows()*ex.cols()<<" elements stored in "<<fsize_bytes<<" bytes"<<endl;
            if(1){ // read it back and assert equivalent data
#if 1
                decltype(ex) fx;
                ifstream ifs(fname);
                eigen_io_bin( ifs, fx );
                ifs.close();
                assert( fx.rows() == ex.rows() );
                assert( fx.cols() == fx.cols() );
                double const diff = (fx-ex).squaredNorm();
                assert( diff < 1.e-8 );
                cout<<"\tGood, read back "<<fname<<" as DenseMf with sumsqr-difference "<<diff<<endl;
#else
                ifstream ifs(fname);
                uint64_t rows,cols;
                io_bin(ifs,rows);
                io_bin(ifs,cols);
                DenseM fx(rows,cols);
                fx.setZero();
                assert( fx.rows() == ex.rows() );
                assert( fx.cols() == fx.cols() );
                Eigen::VectorXf rowr;
                rowr.resize(cols);
                rowr.setZero();
                for(uint64_t r=0U; r<rows; ++r){
                    io_bin( ifs, (void*)rowr.data(), size_t(cols*sizeof(float)) ); // valgrind?
#ifndef NDEBUG
                    double const diff = (rowr-ex.row(r).transpose()).squaredNorm();
                    assert( diff < 1.e-8 );
#endif
                    for(uint64_t c=0U; c<cols; ++c){
                        fx.coeffRef(r,c) = static_cast<double>(rowr.coeff(c));
                    }
                }
                ifs.close();
                cout<<"\tGood, read-back float file --> DenseM (doubles) OK"<<endl;
#endif
            }
        }
        { ostringstream oss; oss<<"mcgen-"<<p.str()<<"-x-S.bin"; fname = oss.str(); }
        { // write x into fname, sparse binary of ** float **
            assert( x.size() > 0 );
            assert( x.size() == y.size() );
            assert( x.size() == p.nxStd + p.nx );
            assert( x[0].size() == p.dim );
            SparseMf ex( x.size(), p.dim );
            {
                // x is plain vector<vector<float>>
                uint64_t nnz=0U;
                for(uint32_t i=0; i<x.size(); ++i)
                    for(uint32_t j=0; j<x[i].size(); ++j)
                        if( static_cast<float>(x[i][j]) != 0.0f )
                            ++nnz;

                typedef Eigen::Triplet<float> T;
                std::vector<T> tripletList;
                tripletList.reserve( nnz );
                for(uint32_t i=0; i<x.size(); ++i)
                    for(uint32_t j=0; j<x[i].size(); ++j)
                        if( static_cast<float>(x[i][j]) != 0.0f )
                            tripletList.push_back( T(i,j,x[i][j]) );

                ex.setFromTriplets( tripletList.begin(), tripletList.end() );
            }
            {
                cout<<" writing "<<fname<<" ... "; cout.flush();
                ofstream ofs(fname);
                eigen_io_bin( ofs, ex ); // x is Dense
                ofs.close();
                cout<<" OK"<<endl;
            }
            uint64_t fsize_bytes;
            {
                struct stat st;
                stat(fname.c_str(), &st);
                fsize_bytes = st.st_size;         // total size, in bytes
            }
            cout<<" Wrote 'x' file "<<fname<<" [ "<<ex.rows()<<" x "<<ex.cols()<<" ] : "
                <<ex.rows()*ex.cols()<<" elements stored in "<<fsize_bytes<<" bytes"<<endl;
            cout<<" ex:\n"; dump(cout,ex);  // no values, outer/inner index lists only
            //cout<<" ex:\n"<<ex<<endl;   // with values
            if(1){ // read it back and assert equivalent data
                SparseMf fx;
                ifstream ifs(fname);
                eigen_io_bin( ifs, fx );
                ifs.close();
                assert( fx.rows() == ex.rows() );
                assert( fx.cols() == fx.cols() );
                double const diff = (fx-ex).squaredNorm();
                assert( diff < 1.e-8 );
                cout<<"\tGood, read back "<<fname<<" as SparseMf with sumsqr-difference "<<diff<<endl;
            }
            if(1){ // read it back with conversion to SparseM
                // This should be AUTOMATIC  -- it is now, for SparseM, but not yet for DenseM
                SparseM fx;
                ifstream ifs(fname);
                eigen_io_bin( ifs, fx );
                ifs.close();
                assert( fx.rows() == ex.rows() );
                assert( fx.cols() == ex.cols() );
                assert( fx.data().size() == ex.data().size() );
                for(uint32_t o=0U; o<fx.outerSize(); ++o){
                    assert( fx.outerIndexPtr()[o] == ex.outerIndexPtr()[o] );
                }
                for(uint32_t i=0U; i<fx.innerSize() + 1 ; ++i){
                    assert( fx.innerIndexPtr()[i] == ex.innerIndexPtr()[i] );
                }
                for(uint32_t d=0U; d<fx.data().size(); ++d){
                    assert( static_cast<float>(fx.valuePtr()[d]) == static_cast<float>(ex.valuePtr()[d]) );
                }
                cout<<"\tGood, read-back float file -->SparseM (doubles) OK"<<endl;
            }
        }
        { ostringstream oss; oss<<"mcgen-"<<p.str()<<"-slc-y.bin"; fname = oss.str(); }
        {
            SparseMb sy( y.size(), p.nClass );  // eigen sparse-y
            {
                sy.resizeNonZeros(y.size());
                // y is plain vector<uint32_t>, single label per example
                assert( y.size() == p.nxStd + p.nx );
                typedef Eigen::Triplet<bool> T;
                std::vector<T> tripletList;
                tripletList.reserve( y.size() );
                for(uint32_t r=0U; r<y.size(); ++r){
                    assert( y[r] < p.nClass );
                    tripletList.push_back( T(r,y[r],true) );
                }
                sy.setFromTriplets( tripletList.begin(), tripletList.end() );
                struct KeepTrue {
                    bool operator()( SparseMb::Index const&, SparseMb::Index const&, SparseMb::Scalar const value ) const
                    {
                        return value==true;
                    }
                };
                sy.prune( KeepTrue() );
                sy.makeCompressed();
            }
            {
                ofstream ofs(fname);
                //eigen_io_bin( ofs, sy ); // y is sparse bool
                // 32 nonzero elements stored in 672 bytes  <--- FIXME (printing.hh, eigen_io_bin)
                // --> 32 nonzero elements stored in 441 bytes  (smaller outerindex)
                // --> 32 nonzero elements stored in 217 bytes  (smaller innerindex)
                // --> 32 nonzero elements stored in 113 bytes  (values as boost::dynamic_bitset)
                // --- now get rid of useless '1' if ALL the bits are set (bool optimization)
                eigen_io_binbool( ofs, sy ); // y is sparse bool, compressed, only-true : 93 bytes
                ofs.close();
            }
            uint64_t fsize_bytes;
            {
                struct stat st;
                stat(fname.c_str(), &st);
                fsize_bytes = st.st_size;         // total size, in bytes
            }
            cout<<" Wrote 'y' file "<<fname<<" [ "<<sy.rows()<<" x "<<sy.cols()<<" ] : "
                //<<sy.outerIndexPtr()[sy.outerSize()]
                <<"SparseMb with "<<sy.nonZeros()
                <<" nonzero elements stored in "<<fsize_bytes<<" bytes"
                <<"\n(OUCH!)\n\n"
                <<endl;
            if(1){ // read it back and assert equivalent data
                SparseMb fy;
                ifstream ifs(fname);
                //std::array<char,4> magic = {'S', 'M', 'b', 'b' };
                //io_bin( ifs, magic );
                //cout<<" magic "<<magic[0]<<magic[1]<<magic[2]<<magic[3]; cout.flush();
                eigen_io_binbool( ifs, fy );
                ifs.close();
                cout<<"sy "; dump(cout,sy);     // short output
                cout<<"fy "; dump(cout,fy);
                assert( fy.isCompressed() );
                assert( fy.rows() == sy.rows() );
                assert( fy.cols() == sy.cols() );
                assert( fy.data().size() == sy.data().size() );
                for(uint32_t r=0U; r<sy.rows(); ++r){
                    assert( sy.outerIndexPtr()[r+1] == sy.outerIndexPtr()[r] + 1 );
                }
                for(uint32_t o=0U; o<fy.outerSize(); ++o){
                    assert( fy.outerIndexPtr()[o] == sy.outerIndexPtr()[o] );
                }
                for(uint32_t i=0U; i<fy.innerSize() + 1 ; ++i){
                    assert( fy.innerIndexPtr()[i] == sy.innerIndexPtr()[i] );
                }
#if 0 // now we use eigen_io_binbool, so all values are 'true'
                for(uint32_t d=0U; d<fy.data().size(); ++d){
                    assert( fy.valuePtr()[d] == sy.valuePtr()[d] );
                }
                double const diff = (fy-sy).squaredNorm();
                assert( diff == 0.0 );
#endif
                cout<<"\tGood, read back "<<fname<<" as SparseMb OK"<<endl;
            }
        }
        { ostringstream oss; oss<<"mcgen-"<<p.str()<<"-mlc-y.bin"; fname = oss.str(); }
        {
            SparseMb sy( y.size(), p.nClass );  // eigen sparse-y, mlc this time (ymap)
            {
                assert( ymap.size() == p.nClass );
                assert( y.size() == p.nxStd + p.nx );
                // ymap[cl] is vector<uint32_t>, multiple labels per example
                typedef Eigen::Triplet<bool> T;
                std::vector<T> tripletList;
                uint64_t nnz = 0U;      for(auto const& cl: y){ nnz+=ymap[cl].size(); }
                tripletList.reserve( nnz );
                for(size_t i=0U; i<y.size(); ++i){
                    for(auto const cls: ymap[y[i]]){
                        assert( cls < p.nClass );
                        tripletList.push_back( T(i,cls,true) );
                    }
                }
                sy.resizeNonZeros(nnz);
                sy.setFromTriplets( tripletList.begin(), tripletList.end() );
                // if all bool 0's removed, then don't need to store values at all XXX
                struct KeepTrue {
                    bool operator()( SparseMb::Index const&, SparseMb::Index const&, SparseMb::Scalar const value ) const
                    {
                        return value==true;
                    }
                };
                sy.prune( KeepTrue() );
                //sy.prune( 1 );  // simpler equiv // but doesn't work for bool
                sy.makeCompressed();
            }
            {
                ofstream ofs(fname);
                //eigen_io_bin( ofs, sy ); // y is sparse bool
                // 64 nonzero elements stored in 1056 bytes !!!
                // --> 64 nonzero elements stored in 825 bytes
                // --> 64 nonzero elements stored in 377 bytes
                // --> 64 nonzero elements stored in 145 bytes
                eigen_io_binbool( ofs, sy ); // y sparse, compressed, only-true: 125 bytes
                ofs.close();
            }
            uint64_t fsize_bytes;
            {
                struct stat st;
                stat(fname.c_str(), &st);
                fsize_bytes = st.st_size;         // total size, in bytes
            }
            cout<<" Wrote 'y' file "<<fname<<" [ "<<sy.rows()<<" x "<<sy.cols()<<" ] : "
                //<<sy.outerIndexPtr()[sy.outerSize()]
                <<"SparseMb with "<<sy.nonZeros()
                <<" nonzero elements stored in "<<fsize_bytes<<" bytes"
                <<"\n(OUCH!)\n\n"
                <<endl;
            if(1){ // read it back and assert equivalent data
                SparseMb fy;
                ifstream ifs(fname);
                eigen_io_binbool( ifs, fy );
                ifs.close();
                cout<<"sy "; dump(cout,sy);     // short output
                cout<<"fy "; dump(cout,fy);
                assert( fy.isCompressed() );
                assert( fy.rows() == sy.rows() );
                assert( fy.cols() == sy.cols() );
                assert( fy.data().size() == sy.data().size() );
                for(uint32_t r=0U; r<sy.rows(); ++r){ // only if ymap[i].size() is always 2...
                    assert( sy.outerIndexPtr()[r+1] == sy.outerIndexPtr()[r] + 2 );
                }
                for(uint32_t o=0U; o<fy.outerSize(); ++o){
                    assert( fy.outerIndexPtr()[o] == sy.outerIndexPtr()[o] );
                }
                for(uint32_t i=0U; i<fy.innerSize() + 1 ; ++i){
                    assert( fy.innerIndexPtr()[i] == sy.innerIndexPtr()[i] );
                }
#if 0 // skip the "all-true" values
                for(uint32_t d=0U; d<fy.data().size(); ++d){
                    assert( fy.valuePtr()[d] == sy.valuePtr()[d] );
                }
                double const diff = (fy-sy).squaredNorm();
                assert( diff == 0.0 );
                cout<<"\tGood, read back "<<fname<<" as SparseMb OK, diff="<<diff<<endl;
#endif
                cout<<"\tGood, read back "<<fname<<" as SparseMb OK"<<endl;
            }
        }
    }
#endif
    // 10b. generate a usable MCfilter ".soln" file with JUST weights of the 'ideal' solution
    //     --> mcgen-a3-txt.soln  and mcgen-a3-bin.soln
    // and --> mcgen-a3-mlc-txt.soln  and mcgen-a3-mlc-bin.soln
    //
    //     NOTE: the mlc .soln, for this ymap, is NOT perfectly separable.
    //     In fact,
    //          ./mcproj --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin
    //     generates 4 possible classes for each example,
    //     instead of the 2 classes in ymap
    //
    //     It may be possible to perfectly separate with 4 projecting lines ??? 
    {
#if ! USE_LIBMCFILTER
        cout<<" Not linked with libmcfilter, NO .soln file output"<<endl;
#else
        assert( soln.size() == p.axes );
        assert( x[0].size() == p.dim );
        cout<<" Forming MCsoln (as Eigen column matrices) ..."<<endl;
        MCsoln mcs;
        mcs.d           = p.dim;
        mcs.nProj       = p.axes;
        mcs.nClass      = p.nClass;     // NO class remap
        //mcs.fname       =
        { // copy vector<vector<float>> weights ---> Eigen DenseM MCsoln::weights_avg
            mcs.parms.no_projections = mcs.nProj;
            // TODO XXX The following can be played with until something that stays at
            // the soln gets an appropriate norm for w, and margins respecting C1 and C2
            //mcs.parms.C2 = mcs.parms.C1 / mcs.nClass;
            //  Note: C1 ~ keep
            mcs.C2 = mcs.parms.C2 = mcs.nClass * p.axes / (p.axes+1U);   // heuristic [omit/modify at will]
            mcs.C1 = mcs.parms.C1 = mcs.C2 * mcs.nClass;
            //mcs.parms.optimizeLU_epoch = 100U;
            mcs.parms.batch_size = 100U;
            mcs.parms.max_iter = 2000U;
            //mcs.parms.reoptimize_LU = true;

            mcs.weights_avg.conservativeResize( mcs.d, mcs.nProj );     // soln, as col. vectors
            double wnorm = 1.0; // (p.axes + 1U);                                 // <-- NEW
            for(uint32_t d=0U; d<mcs.d; ++d){
                for(uint32_t s=0U; s<mcs.nProj; ++s){
                    mcs.weights_avg.coeffRef(d,s) = soln[s][d] * wnorm;
                }
            }
            cout<<"\tw["<<mcs.weights_avg.rows()<<","<<mcs.weights_avg.cols()<<"]\n"
                <<mcs.weights_avg<<endl;
            assert( mcs.weights_avg.rows() == mcs.d );
            assert( mcs.weights_avg.cols() == mcs.nProj );
        }
        for(uint32_t slc_mlc=0U; slc_mlc<2U; ++slc_mlc)
        { // sample code to find {l,u} bounds for fully separable case, and print
            // to find {l,u} bounds of each soln, using FULL set of margin points, with strict +ve margin
            DenseM & l = mcs.lower_bounds_avg;
            DenseM & u = mcs.upper_bounds_avg;
            l.conservativeResize(mcs.nClass, mcs.nProj);
            u.conservativeResize(mcs.nClass, mcs.nProj);
            for(int i=0U; i<mcs.nClass; ++i){
                for(uint32_t p=0U; p<mcs.nProj; ++p){
                    l.coeffRef(i,p) = numeric_limits<double>::max();
                    u.coeffRef(i,p) = numeric_limits<double>::min();
                }
            }
            {// Form ACTUAL {l,u} bounds of training examples [p.nx x p.dim]
                // (last time was for trivial soln and idealized margin pushpoint)
                // TODO XXX Best: project/rot/skew/embed the idealized margin-expansions XXX
                // (but that is a lot more typing)
                // soln will still be good if |m|=a in parms, because the idealized margin-expansions
                // will always be in the training data, 'x'.
                for(uint32_t i=0U; i<x.size(); ++i){
                    auto const& v = x[i];
                    if( slc_mlc == 0U ){            // y[i] --> slc {l,u} bounds
                        auto cls = y[i];
                        assert( v.size() == p.dim );
                        assert( cls < mcs.nClass );
                        for(uint32_t p=0U; p<mcs.nProj; ++p){     // for each soln unit vector
                            float const vdots = dot( v, soln[p] );
                            l.coeffRef(cls,p) = min( static_cast<float>(l.coeff(cls,p)), vdots );    // update l
                            u.coeffRef(cls,p) = max( static_cast<float>(u.coeff(cls,p)), vdots );    // and u bounds
                        }
                    }else{                          // ymap[y[i]] --> mlc {l,u} bounds
                        for(auto const cls: ymap[ y[i] ] ){ // multi-label
                            //auto v = m.mid; // OHOH: this ideal split-point has only p.axes dims
                            assert( v.size() == p.dim );
                            assert( cls < mcs.nClass );
                            for(uint32_t p=0U; p<mcs.nProj; ++p){     // for each soln unit vector
                                float const vdots = dot( v, soln[p] );
                                l.coeffRef(cls,p) = min( static_cast<float>(l.coeff(cls,p)), vdots );    // update l
                                u.coeffRef(cls,p) = max( static_cast<float>(u.coeff(cls,p)), vdots );    // and u bounds
                            }
                        }
                    }
                    // Now push apart the {l,u} bounds by a slight bit (non-zero margin)
                    //    see comments in Filter.h about bad things with zero-margin {l,u} !
                    l.array() -= 1.1111e-4;
                    u.array() += 1.1111e-4;
                    if(1){ //print, you can very that every l,u pairs is shattered when all solns considered
                        cout<<" {l,u} solns from Eigen col(p).transpose() ...:";
                        for(uint32_t p=0U; p<soln.size(); ++p){
                            cout<<"\n\tl["<<p<<"] = "<<l.col(p).transpose();
                            cout<<"\n\tu["<<p<<"] = "<<u.col(p).transpose();
                        }
                        cout<<endl;
                    }
                    if(1){
                        cout<<"\n *** Final MCsoln to save ***"<<endl;
                        mcs.pretty(cout);
                    }
                }
            }//end forming {l,u} bounds
            string fnameSolnBase;
            {
                stringstream oss;
                oss<<"mcgen-"<<p.str()<<(slc_mlc==0U?"":"-mlc");
                fnameSolnBase = oss.str();
            }
            assert( mcs.weights_avg.rows() == mcs.d );
            assert( mcs.weights_avg.cols() == mcs.nProj );
            assert( mcs.lower_bounds_avg.rows() == mcs.nClass );
            assert( mcs.lower_bounds_avg.cols() == mcs.nProj );
            assert( mcs.upper_bounds_avg.rows() == mcs.nClass );
            assert( mcs.upper_bounds_avg.cols() == mcs.nProj );
            mcSave( fnameSolnBase, mcs );
        }// slc_mlc
#endif //USE_LIBMCFILTER
    }

    cout<<"\nGoodbye"<<endl;
}

