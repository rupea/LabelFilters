
#include "evaluate.hh"

void output_perfs(const std::vector<double>& MicroF1, const std::vector<double>&  MacroF1,
		  const std::vector<double>& MacroF1_2, 
		  const std::vector<double>& MicroPrecision,
		  const std::vector<double>& MacroPrecision, 
		  const std::vector<double>& MicroRecall, 
		  const std::vector<double>& MacroRecall,
		  const std::vector<double>& Top1, const std::vector<double>& Top5,
		  const std::vector<double>& Top10, const std::vector<double>& Prec1, 
		  const std::vector<double>& Prec5, const std::vector<double>& Prec10,
		  const VectorXsz& nact, size_t total_preds, 
		  const std::vector<double>& total_time, 
		  const std::vector<double>& filter_time, 
		  const std::vector<double>& predict_time, 
		  string str/*=""*/, ostream& out/*=cout*/)
{
  int proj;
  if (MicroF1.size() > 1)
    {
      out << str << "MicroF1_per_proj  ";
      for (proj = MicroF1.size() - 1; proj >=0 ; proj--)
	{
	  out << MicroF1[proj] << "  ";
	}
      out << endl;
    }
  if (MacroF1.size() > 1)
    {
      out << str << "MacroF1_per_proj  ";
      for (proj = MacroF1.size()-1;proj >=0 ; proj--)
	{
	  out << MacroF1[proj] << "  ";
	}
      out << endl;
    }
  if (MacroF1_2.size() > 1)
    {
      out << str << "MacroF1_2_per_proj  ";
      for (proj = MacroF1_2.size()-1;proj >=0 ; proj--)
	{
	  out << MacroF1_2[proj] << "  ";
	}
      out << endl;
    }
  if (MicroPrecision.size() > 1)
    {
      out << str << "MicroPrecision_per_proj  ";
      for (proj = MicroPrecision.size()-1; proj >=0; proj--)
	{
	  out << MicroPrecision[proj] << "  ";
	}
      out << endl;
    }
  if (MacroPrecision.size() > 1)
    {
      out << str << "MacroPrecision_per_proj  ";
      for (proj = MacroPrecision.size()-1;proj >=0 ; proj--)
	{
	  out << MacroPrecision[proj] << "  ";
	}
      out << endl;
    }
  if (MicroRecall.size() > 1)
    {
      out << str << "MicroRecall_per_proj  ";
      for (proj = MicroRecall.size()-1;proj >=0 ; proj--)
	{
	  out << MicroRecall[proj] << "  ";
	}
      out << endl;
    }
  if (MacroRecall.size() > 1)
    {
      out << str << "MacroRecall_per_proj  ";
      for (proj = MacroRecall.size()-1;proj >=0 ; proj--)
	{
	  out << MacroRecall[proj] << "  ";
	}
      out << endl;
    }
  if (Top1.size() > 1)
    {
      out << str << "Top1_per_proj  ";
      for (proj = Top1.size()-1;proj >=0 ; proj--)
	{
	  out << Top1[proj] << "  ";
	}
      out << endl;
    }
  if (Prec1.size() > 1)
    {
      out << str << "Prec1_per_proj  ";
      for (proj = Prec1.size()-1;proj >=0 ; proj--)
	{
	  out << Prec1[proj] << "  ";
	}
      out << endl;
    }
  if (Top5.size() > 1)
    {
      out << str << "Top5_per_proj  ";
      for (proj = Top5.size()-1;proj >=0 ; proj--)
	{
	  out << Top5[proj] << "  ";
	}
      out << endl;
    }
  if (Prec5.size() > 1)
    {
      out << str << "Prec5_per_proj  ";
      for (proj = Prec5.size()-1;proj >=0 ; proj--)
	{
	  out << Prec5[proj] << "  ";
	}
      out << endl;
    }
  if (Top10.size() > 1)
    {
      out << str << "Top10_per_proj  ";
      for (proj = Top10.size()-1;proj >=0 ; proj--)
	{
	  out << Top10[proj] << "  ";
	}
      out << endl;
    }
  if (Prec10.size() > 1)
    {
      out << str << "Prec10_per_proj  ";
      for (proj = Prec10.size()-1;proj >=0 ; proj--)
	{
	  out << Prec10[proj] << "  ";
	}
      out << endl;
    }

  // the number of active classes is always calculated
  out << str << "active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << nact[proj] << "  ";
    }
  out << endl;
  out << str << "prc_active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << nact[proj]*1.0/total_preds << "  ";
    }
  out << endl;
  out << str << "speedup_active_per_proj  ";
  for (proj = nact.size()-1; proj >=0 ; proj--)
    {
      out  << total_preds*1.0/nact[proj] << "  ";
    }
  out << endl;

  out << str << "MicroF1  " << MicroF1[0] << endl;
  out << str << "MacroF1  " << MacroF1[0] << endl;
  out << str << "MacroF1_2  " << MacroF1_2[0] << endl;
  out << str << "MicroPrecision  " << MicroPrecision[0] << endl;
  out << str << "MacroPrecision  " << MacroPrecision[0] << endl;
  out << str << "MicroRecall  " << MicroRecall[0] << endl;
  out << str << "MacroRecall  " << MacroRecall[0] << endl;
  out << str << "Top1  " << Top1[0] << endl;
  out << str << "Prec1  " << Prec1[0] << endl;
  out << str << "Top5  " << Top5[0] << endl;
  out << str << "Prec5  " << Prec5[0] << endl;
  out << str << "Top10  " << Top10[0] << endl;
  out << str << "Prec10  " << Prec10[0] << endl;
  out << str << "nactive  " << nact(0) << endl;
  out << str << "prc_active  " << nact(0)*1.0/total_preds << endl;
  out << str << "speedup  " << total_preds*1.0/nact(0) << endl;
  out << str << "total  " << total_preds << endl;
  out << str << "total_time  " << total_time[0] << endl;
  out << str << "filter_time  " << filter_time[0] << endl;
  out << str << "predict_time  " << predict_time[0] << endl;
}

// -------- force some template instantiations --------
// SparseM ...

template
void predict_chunk(predvec& predictions, VectorXsz& no_active,
		   doublevec& filter_time, doublevec& predict_time,
		   doublevec& total_time,
		   const SparseM& x, 
		   const DenseColM* wmat, const DenseColM* lmat,
		   const DenseColM* umat,		   
		   const DenseColMf& ovaW_chunk, 
		   const size_t start_class, predtype thresh, int k,
		   bool allproj, bool verbose);
template
void evaluate_projection(const std::vector<SparseM*>& x, 
			 const std::vector<SparseMb*>& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 const std::vector<std::string>& setnames,
			 bool allproj, bool verbose, ostream& out = cout);

template
void get_projection_measures(const SparseM& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue);
template
void evaluate_projection_chunks(const std::vector<SparseM*>& x, 
				const std::vector<SparseMb*>& y, 
				const string& ova_file, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				const std::vector<std::string>& setnames,
				bool allproj, bool verbose, ostream& out = cout);

// DenseM ...
template
void predict_chunk(predvec& predictions, VectorXsz& no_active,
		   doublevec& filter_time, doublevec& predict_time,
		   doublevec& total_time,
		   const DenseM& x, 
		   const DenseColM* wmat, const DenseColM* lmat,
		   const DenseColM* umat,		   
		   const DenseColMf& ovaW_chunk, 
		   const size_t start_class, predtype thresh, int k,
		   bool allproj, bool verbose);
template
void evaluate_projection(const std::vector<DenseM*>& x, 
			 const std::vector<SparseMb*>& y, 
			 const DenseColMf& ovaW,
			 const DenseColM* wmat, const DenseColM* lmat,
			 const DenseColM* umat,
			 predtype thresh, int k, const string& projname,
			 const std::vector<std::string>& setnames,
			 bool allproj, bool verbose, ostream& out = cout);
template
void get_projection_measures(const DenseM& x, const SparseMb& y,
			     const DenseColM& wmat, const DenseColM& lmat,
			     const DenseColM& umat, bool verbose,
			     VectorXsz& nrTrueActive, VectorXsz& nrActive, VectorXsz& nrTrue);
template
void evaluate_projection_chunks(const std::vector<DenseM*>& x, 
				const std::vector<SparseMb*>& y, 
				const string& ova_file, int chunks,
				const DenseColM* wmat, const DenseColM* lmat,
				const DenseColM* umat,
				predtype thresh, int k, const string& projname,
				const std::vector<std::string>& setnames,
				bool allproj, bool verbose, ostream& out = cout);

