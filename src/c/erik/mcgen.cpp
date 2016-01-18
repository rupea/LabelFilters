/** \file
 * test case generator for multi-class label filter.
 */

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include <cmath>        // acos

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
        void randCube( Vector& v);      ///< fill v[] with values in [0,+1]
        void randDirn( Vector& v);      ///< fill v[] with a rand unit vector (seq rotns alg)
        Vector&& randCube(uint32_t sz); ///< resize and fill v[] with values in [0,+1]
        Vector&& randDirn(uint32_t sz); ///< resize and fill v[] with a rand unit vector (seq rotns alg)
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
            Parms();                        ///< construct with defaults, and call init()

            /// \name core settings
            //@{
            uint32_t axes;                  ///< [3] # projection axes to separate examples
            bool     ball;                  ///< [*] true = examples on unit ball, false = rotated unit hypercube
            uint32_t dim;                   ///< [3] dimension of each training example
            int32_t  margin;                ///< [+axes] |margin|<=a 0=none, +ve=separable, -ve=non-separable
            uint32_t parts;                 ///< [2] each axis separates into parts, nClass = parts ^ axes [2^3]
            bool     trivial;               ///< [1]: soln is first \c axes unitvecs (skew,rot=0), 0: use skew,rot!=0
            uint32_t skew;                  ///< [0] skew < axes dims toward (1,0,0,...) of trivial soln
            uint32_t rot;                   ///< [0] then project into axes < rot < dim and do a random rotation
            //@}
            /// \name non-core settings set via init()
            //@{
            uint32_t nClass;                ///< = parts ^ axes
            uint32_t nxStd;                 ///< pre-determined examples 1 per nClass + (margin!=0? 2*|margin|*a*(parts-1)
            uint32_t nx;                    ///< default (and silent minimum) is max(4*axes, nxStd)
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
        //void randCube( Vector& v);     ///< fill v[] with values in [-1,+1]
        //void randDirn( Vector& v);     ///< fill v[] with a rand unit vector (seq rotns alg)
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
              , ball( true )
              , dim( 3U )
              , margin( 3U )
              , parts( 2U )
              , trivial( true )
              , skew( 0U )
              , rot( 0U )
              , nClass( 0U ) // set via init()
              , nxStd( 0U )     // set via init()
              , nx( 0U )        // set via init()
        {
            //init();
        }
        void Parms::init(){
            int const verbose=0;
            nClass = parts;
            if(verbose) {cout<<" nClass="<<parts; cout.flush();}
            for(uint32_t i=1U; i<axes; ++i){
                nClass *= parts;
                if(verbose){cout<<" .."<<i<<".."<<nClass; cout.flush();}
            }
            if(verbose) cout<<endl;
            nxStd = nClass;
            if(verbose){cout<<" nxStd=nClass="<<nxStd; cout.flush();}
            if( margin != 0 ){
                uint32_t m = static_cast<uint32_t>((margin>0? margin: -margin));
                nxStd += 2 * m *axes * (parts-1U);
                if(verbose){cout<<" += [2|m|a(p-1)=2*|"<<margin<<"|*"<<axes<<"*("<<parts-1U<<")] --> "<<nxStd; cout.flush();}
            }
            uint32_t nxMinimum = std::max(4*axes, 2*nxStd);
            nx = std::max( nx, nxMinimum );                     // nx must be at least some minimum value
            // - 1st of nClass nxStd examples are ~ [farthest] class centers
            // - rest of nxStd are margin-tightening examples
            // - remaining (nx-nxStd) randomly scattered over example space.
            if(verbose) cout<<endl;
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

                ("axes,a", po::value<uint32_t>()->implicit_value(3U)
                 , "[3] projection|soln axes, 2 <= a <= d")
                ("ball,b", po::value<bool>()->implicit_value(true)->default_value(true)
                 , "[*] examples on unit ball")
                ("cube,c", po::value<bool>()->implicit_value(false)->default_value(false)
                 , "... or examples in [deformed? r,s] unit cube")
                ("dim,d", po::value<uint32_t>()->implicit_value(3U)
                 , "[3] dimension of training data, d >= 2")
                ("margin,m", po::value<int32_t>()
                 , "[a] |m|<=a, add 2|m|a(p-1) margin tightness examples."
                 "  +ve: separable, -ve: non-separable, 0: no margin-examples"
                 )
                ("parts,p", po::value<uint32_t>()->implicit_value(2U)->default_value(2U)
                 , "[2] p>=2 parts to divide each -a axis into. # classes = p^a [8]")

                ("rot,r", po::value<uint32_t>()
                 , "after t+s, move into a<=r<=d dims & rotate randomly (keep lengths,angles)")
                ("skew,s", po::value<uint32_t>()
                 , "after t, rot 0<s<a-1 dims (2..s+1) 1st unit (1,0,0...) (axes-->non-orthog)")
                ("trivial,t"
                 , "[*] soln: unit vectors in 1st -a dimensions form projection axes")

                ("examples,x", po::value<uint32_t>()
                 , "[0] How many, silently adjusted up to max(a*4, 2*<pre-determined examples>")
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
                "\n - s skewed signal still in 1st a dims"
                "\n - r moves signal into 1st a<=r<=d dims, higher dims ~ noise"
                ;
        }
        /** \c p might not reflect values it would have after init() */
        std::string Parms::str(){
            Parms def;          // get default core values (well, some have been inited to nonzero)
            ostringstream s;
            s<<'a'<<axes;                               // non-optional
            if( ball != def.ball ) s<<(ball? 'b': 'c'); // optional
            s<<'d'<<dim;                                // non-optional
            if( margin != static_cast<int32_t>(axes) ) s<<'m'<<margin;
            //if( parts != def.parts ) s<<'p'<<parts;     // optional
            s<<'p'<<parts;                              // non-optional
            if( ! trivial ){
                assert( skew > 0U || rot > 0U );
                if( skew != 0U ) s<<'s'<<skew;
                if( rot != 0U ) s<<'r'<<rot;
            }
            // nx is tricky: it is non-core, but default value is set by init()
            def = *this;
            def.init();          // default value is now cp.nx
            if( nx != def.nx && nx != 0U ) s<<'x'<<def.nx;
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

                po::notify(vm);

                if( vm.count("help") ) {
                    helpUsage( cout );
                    cout<<desc<<endl;
                    helpExamples(cout);
                    return;
                }
                if( vm.count("axes") ) { parms.axes = vm["axes"].as<uint32_t>(); }
                if( vm.count("dim") ) { parms.dim = vm["dim"].as<uint32_t>(); }
                if( vm.count("margin") ) {
                    parms.margin = vm["margin"].as<int32_t>();
                    cout<<" parms.margin --> "<<parms.margin<<endl;
                }
#if 1
                if( vm.count("parts") ) { parms.parts = vm["parts"].as<uint32_t>(); }
                //if( vm.count("trivial") ) {
                //    int32_t const t = vm["trivial"].as<int32_t>();
                //    if( t != 0 && t != 1 ) throw runtime_error("-t must be zero or 1");
                //    parms.trivial = static_cast<bool>(t);
                //}
                if( vm.count("skew") ) {
                    parms.skew = vm["skew"].as<uint32_t>();
                }
                if( vm.count("rot") ) {
                    parms.rot = vm["rot"].as<uint32_t>();
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
#endif
#if 0
                opt.time = 0U;          // replace NWRITES
                opt.traces = 0U;        // use traces unless other args (--time) force us not to
                opt.awPercent = 0U;
                opt.ek23 = vm.count("ek23");
                opt.tp25 = vm.count("tp25");
                opt.idmap = vm.count("idmap");

                if( vm.count("time") )  // if --time (or -t) option appeared, only OK for fake workload
                {
                    opt.time = vm["time"].as<float>()*SECONDSTOTIME;
                    cout<<" opt.time represents "<<opt.time*TIMETOSECONDS<<" s"<<endl;
                    //if( vm.count("traces") && vm["traces"].as<bool>() )
                    //{
                    //cout<<" our TracePlay doesn't know how to shut off after a given --time="
                    //cout<<" our TracePlay MIGHT know how to shut off after a given --time="
                    //    <<opt.time*TIMETOSECONDS<<endl;
                    //throw "oops";
                    //}
                    //opt.traces = 0;
                }
                if( vm.count("async") )  // if --time (or -t) option appeared, only OK for fake workload
                {
                    opt.awPercent = vm["async"].as<uint32_t>();
                    if( opt.awPercent > 100U )
                    {
                        cout<<" --async="<<opt.awPercent<<" \%-of-writes-async must be in range [0,100]"<<endl;
                        throw("oops");
                    }
                }

                opt.nTimes = vm["times"].as<uint32_t>();
                opt.tMon = vm["tMon"].as<float>();

                opt.wol  = vm["wol"].as<uint32_t>();
                if( opt.wol > 3U )
                {
                    cout<<" --wol="<<opt.wol<<" out of range [0,3]"<<endl;
                    throw " --wol should be 0,1,2,3";
                }
                if( opt.swap == 0U && opt.wol == 1U )
                {
                    cout<<"--wol ineffective with --swap=0 (NEVERSWAP)"<<endl;
                    throw("oops");
                }

                opt.nio = 0U;
                if( vm.count("nio") )   // --nio (or -n) option appeared, only OK for trace files
                {
                    opt.nio = vm["nio"].as<uint32_t>(); 
                    // BOTH WorkModel and current TwrkWrite DO accept --nio switch
                }

                if( vm.count("traces") )        // if --traces (or -T) appeared as an option...
                {
                    opt.traces = vm["traces"].as<bool>();
                }
                { // --traces with missing --time / --nio specs get some low default values
                    if(  opt.traces && opt.nio == 0U )
                    {
                        //opt.nio = 20U;   // something small
                        //cout<<" TracePlay will use low default value of --nio="<<opt.nio<<endl;
                        cout<<" pa21 using whole trace files, --nio="<<opt.nio<<endl;
                    }
                    if( !opt.traces && opt.time == 0U && opt.nio == 0U )
                    {
                        opt.time = 1U*SECONDSTOTIME;   // something small
                        cout<<" WorkModel will use low default value of --time="
                            <<opt.time*TIMETOSECONDS<<" (seconds)"<<endl;
                    }
                    // TODO -- fix loadGen.cpp so that --time=0 makes it go "forever". for now kludge it
                    //         by using some high value (e.g. 50000 seconds)
                    if( !opt.traces && opt.nio > 0U && opt.time == 0U )
                    {
                        opt.time = 50000U * SECONDSTOTIME;
                        cout<<" WorkModel with --nio will use some high-ish value --time="
                            <<opt.time*TIMETOSECONDS<<" (seconds)"<<endl;
                    }
                }

                // ... ADD MORE OPTIONS HERE ...

                // ... check for more unsupported combinations? ...

                if( vm.count("dryrun") )
                {
                    cout<<asOptions( opt )<<endl;
                    exit(0);
                }
#endif
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
            return;
        }
    }//opt::


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

using namespace std;
using namespace mcgen;
int main(int argc, char** argv)
{
    opt::Parms p;
    cout<<" default canonical args: "<<p.str()
        <<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    opt::argsParse( argc, argv, p );
    string canonicalArgs = p.str();
    cout<<"         canonical args: "<<canonicalArgs
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
    p.init();
    // 1. generate the trivial axis solution
    // 2. generate trivial solutions: 1 central example per class
    // 3. generate trivial solutions: according the p.margin
    // 4. generate random examples (retry if any fail +ve p.margin)
    // 5. generate labels (use non-native label if in -ve margin region)
    // 6. generate any skew/rot transformation data
    // 7. apply transforms to each of p.axes projection axes, write trainFile
    // 8. apply transforms to each training examples, write axesFile
    // [9. generate a usable MCsoln file with the 'ideal' solution]
    cout<<"\nGoodbye"<<endl;
}

