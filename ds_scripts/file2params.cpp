#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
  std::istream *pin;
  std::ifstream in;
  std::string line;
  std::string param;
  std::string s_value;
  double d_value;

  if (argc == 2)
    {
      in.open(argv[1]);
      pin = &in;
    }						
  else
    {
      pin = &cin;
    }

  int l = 1;
  cout << "{";
  while (!pin->eof())
    {     
      getline(*pin,line);
      param.clear();
      s_value.clear();
      d_value = 0;
      std::stringstream sl(line);
      sl >> param >> s_value;
      if (param.length() == 0 )
	{
	  // must have been an empty line
	  continue;
	}
      if (s_value.length() == 0)
	{
	  cerr << "ERROR parsing line " << l<< endl;
	  exit(-1);
	}
      
      if ( l > 1)
	{
	  cout << ",";
	}
      std::stringstream sv(s_value);
      sv >> d_value;
      if (!sv || !sv.eof())
	{
	  cout << "\"" << param << "\":\"" << s_value << "\"";
	}
      else
	{
	  cout << "\"" << param<< "\":" << d_value;
	}
      l++;
    }
  
  cout << "}" << endl;
  
  if (in.is_open())
    {
      in.close();
    }
  
}
