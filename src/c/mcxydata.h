#ifndef __MCXYDATA_H
#define __MCXYDATA_H

#include "typedefs.h"
#include <array>

/** NEW: introduce a data wrapper.
 *
 * - This allows base x [,y] data to be passed/shared easily between solver
 *   and projector objects, \c MCprojProg and \c MCsolveProgram.
 * - Another benefit is removing duplicate code for reading data.
 * - Eventually, may help move template code into the library? */
class MCxyData {
public:
    MCxyData();
    MCxyData(DenseM const& x);   
    MCxyData(DenseM const& x, SparseMb const& y);
    MCxyData(SparseM const& x);   
    MCxyData(SparseM const& x, SparseMb const& y);
    
    ~MCxyData(){};
    /// \name row-wise test data matrix
    //@{
    // perhaps denseOk and sparseOk can be replaced by xDense.size() != 0 (etc.) ?
    DenseM xDense;
    bool denseOk;
    SparseM xSparse;
    bool sparseOk;
    //@}
    SparseMb y;                 ///< optional for projection operation.
    /// \name optional, private stats
    //@{
private:
    double qscal; ///< if >0, the multiplier used for \c quadx dimensions
    double xscal; ///< if >0, the global x multipler used for \c xscale



public:
    //@}

    void xread( std::string xFile );    ///< read x (binary, sparse/dense) (txt fmt \b todo)
    void yread( std::string yFile );    ///< read x (sparse binary or text)
    void xwrite( std::string xFile ) const; ///< save x (binary only, for now)
    void ywrite( std::string yFile ) const; ///< save y (binary only, for now)

    std::string shortMsg() const;       ///< format+dimensions

    void xunitnormal();   // make x rows into unit vectors by scaling

    /*  remove mean(if center=true),stdev from x cols (colNorm=true) or rows (colNorm=false)(dense only)  */	
    void xstdnormal(bool colNorm = true, bool center = true);  

    /*  remove mean(if center=true),stdev from x cols (colNorm=true) or rows (colNorm=false)(dense only)
    **  return the removed mean and stdev in the mean and stdev vectors. 
    ** if useMeanStdev = true use mean and stdev for normalization rather than the true mean and stdev*/	
    void xstdnormal(Eigen::VectorXd& mean, Eigen::VectorXd& stdev, bool colNorm = true, bool center = true, bool useMeanStdev = false); 

    void xscale(double scal);           ///< multiply all x values by const
    double xmul() const {return xscal;} ///< what's global x multiplier?

    void quadx(double qscal=0.0);       ///< add quadratic dimensions (0.0 autoscales, somehow) \throw if no x data
    double quadmul() const {return qscal;} ///< return the used quadmul (or 0.0 if quadx has not been called)
    
    // I got annoyed with weighting for an error before aborting trying binary
    // reads. So let me (everywhere, sigh) use a magic header, for a quick check.
    static std::array<char,4> magic_xSparse; ///< 0x00,'X','s','8' (or 4 for floats)
    static std::array<char,4> magic_xDense;  ///< 0x00,'X','d','8' (or 4 for floats)
    static std::array<char,4> magic_yBin;    ///< 0x00,'Y','s','b'
    // feel free to add any other [binary] formats.
};

#endif //__MCXYDATA_H
