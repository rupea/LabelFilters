#ifndef R64_H_
#define R64_H_
/** \file r64.h
 * \brief poor-man's replacement for drand48 (fast, \b low-quality random numbers).
 *
 * This is not the full-featured version,
 * which also supports going forward by some arbitrary number of steps.
 *
 * See HydraStore csumMLCG code, and modtable.bc lookup-table generator
 * to add the 'advance(n)' functionality to this generator.
 */
#include <stdint.h>
//namespace sim
//{
    // I'm not too concerned about rng quality, so I'll not use drand48...
    // mlcg2exp64 _multiple and _inverse (LCGADD is any odd number, for now)
    //#define LCGMUL 1181783497276652981ULL
    //#define LCGINV 13515856136758413469ULL
    //#define LCGADD 76543217654321ULL
    //#define LCGMUL2 7664345821815920749ULL
    //#define LCGMUL3 2685821657736338717ULL
    // MUL  MUL2 MUL3  are for m=2^64, c=0, and lack 8/16/32-D correlation
    // MUL4 MUL5 MUL6  are full period for any odd c, decorrelated 8/16/32-tuples
    //#define LCGMUL4 2862933555777941757ULL
    //#define LCGMUL5 3202034522624059733ULL
    //#define LCGMUL6 3935559000370003845ULL
    struct R64
    {
        R64(uint64_t const seed=123U) : r(seed) {}
        R64& operator++()
        {
            this->r = this->r * 2862933555777941757UL //LCGMUL4
                + 13U;
            return *this;
        }
        uint64_t operator()()
        {
            return (this->r = this->r * 2862933555777941757UL //LCGMUL4
                    + 13U);
        }
        uint64_t value() const { return r; }
        uint32_t u32() const { return r ^ (r>>32U); }
        //uint64_t u64() const { return r ^ (r>>32U); }
        uint64_t u64() const { return r; }
        void seed( uint64_t const seed )
        {
            r = seed;
        }
        /** returns 0.0 to 1.0(exclusive).  IEEE double is guaranteed to
         * have at least 53 signicand bits, so we'll base our double value
         * on ~50 bits....
         *
         * (Note that this is much more fine-grained than boost's uniform_01,
         *  but it still would be fastest to JUST twiddle the FP bits directly
         *  in a union{uint64_t;double;}  (but I don't have such twiddles handy)
         *
         * assembler code shows it using xmm registers (uggh!)
         *
         * Note: 50, magic constant, whereas
         *       gcc -dM -E - <<<''
         *       shows a predefined value
         *       #define __DBL_MANT_DIG__ 53
         */
        double drand()
        {
            return double( ((*this)()>>14)
                    * (1.0/(uint64_t(1U)<<50)) // compiler can precalculate this max val
                    );
        }
        private:
        //r50inv = double(1.0)/(uint64_t(1ULL<<50)));
        uint64_t r;
    };
//}//sim::
#endif // R64_H_
