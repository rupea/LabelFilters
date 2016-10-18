#include <iostream>
#include <vector>
#include <string>

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <mex.h>

using namespace std;

#define pairIF pair<int,float>
#define NAME_LEN 100000
#define Malloc(ptr,type,n) ptr = (type*)malloc((n)*sizeof(type))
#define Realloc(ptr,type,n) ptr = (type*)realloc((ptr),(n)*sizeof(type))

int MAX_LEN = 1000000;
static char * line = NULL;
string file_name;

void exit_with_message(string s)
{
  printf("%s\n",s.c_str());
  exit(1);
}

char* read_line(FILE *input)
{
	// reads next line from 'input'. Exits with message on failure


	fgets(line,MAX_LEN,input);
	if(ferror(input))
	{
		char mesg[100];
		sprintf(mesg,"error while reading input file %s\n",file_name.c_str());
		exit_with_message(mesg);
	}
	if(feof(input))
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		MAX_LEN *= 2;
		Realloc(line,char,MAX_LEN);
		int len = (int) strlen(line);
		if(fgets(line+len,MAX_LEN-len,input) == NULL)
			break;
	}

	char * cptr = strrchr(line,'\n');
	*cptr = '\0';

	return line;
}

void parse_line_lbls_fts(vector<int> & labels, vector<pairIF> & features)
{
	int ctr;
	char * tok;
	int id;
	float val;

	char * lbls = line;
	char * lbl_delim = strchr(lbls,' ');
	*lbl_delim = '\0';
	tok = strtok(lbl_delim+1,": ");  

	while(tok)
	{
		id = strtol(tok,NULL,10);
		tok = strtok(NULL,": ");
		val = strtod(tok,NULL);
		features.push_back(make_pair(id,val));
		tok = strtok(NULL,": ");
	}

	tok = strtok(lbls,",");
	while(tok != NULL)
	{
		id = strtol(tok,NULL,10);
		labels.push_back(id);
		tok = strtok(NULL,",");
	}

	return;
}

void read_data(string in_file_name, mxArray*& ft_mat, mxArray*& lbl_mat)
{
	file_name = in_file_name;
	int ctr;
	Malloc(line,char,MAX_LEN);

	int num_inst,num_ft,num_lbl;
	num_inst = num_ft = num_lbl = 0;

	vector<pairIF> fts;
	vector<int> lbls;
	int ft_nnz=0, lbl_nnz=0;

	FILE* data_file;

	data_file = fopen(file_name.c_str(),"r");
	fscanf(data_file,"%d %d %d",&num_inst,&num_ft,&num_lbl);
	//	char c = fgetc(data_file);
	read_line(data_file);
	while(true)
	{
		fts.clear();
		lbls.clear();
		read_line(data_file);
		if(feof(data_file))
			break;

		parse_line_lbls_fts(lbls,fts);

		ft_nnz += fts.size();
		lbl_nnz += lbls.size();
	}
	fclose(data_file);
	ft_mat = mxCreateSparse(num_ft,num_inst,ft_nnz,mxREAL);
	mwIndex* ft_Ir = mxGetIr(ft_mat);
	mwIndex* ft_Jc = mxGetJc(ft_mat);
	ft_Jc[0] = 0;
	double* ft_Pr = mxGetPr(ft_mat);

	lbl_mat = mxCreateSparse(num_lbl,num_inst,lbl_nnz,mxREAL);
	mwIndex* lbl_Ir = mxGetIr(lbl_mat);
	mwIndex* lbl_Jc = mxGetJc(lbl_mat);
	lbl_Jc[0] = 0;
	double* lbl_Pr = mxGetPr(lbl_mat);

	data_file = fopen(file_name.c_str(),"r");
	fscanf(data_file,"%d %d %d",&num_inst,&num_ft,&num_lbl);
	//	c = fgetc(data_file);
	read_line(data_file);
	ctr = 0;
	while(true)
	{
		fts.clear();
		lbls.clear();
		read_line(data_file);
		if(feof(data_file))
			break;

		parse_line_lbls_fts(lbls,fts);

		int siz;

		for(siz=0; siz<fts.size(); siz++)
		{
			ft_Ir[ft_Jc[ctr]+siz] = fts[siz].first;
			ft_Pr[ft_Jc[ctr]+siz] = fts[siz].second;
		}
		ft_Jc[ctr+1] = ft_Jc[ctr]+siz;

		for(siz=0; siz<lbls.size(); siz++)
		{
			lbl_Ir[lbl_Jc[ctr]+siz] = lbls[siz];
			lbl_Pr[lbl_Jc[ctr]+siz] = 1.0;
		}
		lbl_Jc[ctr+1] = lbl_Jc[ctr]+siz;

		ctr++;
	}
	fclose(data_file);

	free(line);
}


void mexFunction(int nlhs, mxArray* plhs[], int rlhs, const mxArray* prhs[])
{
	mxAssert(nrhs==1,"Required and allowed input arguments: text_data_file_name");
	mxAssert(nlhs==2,"Required and allowed input arguments: mex_feature_mat,mex_label_mat");

	char file_name[NAME_LEN];
	mxGetString(prhs[0],file_name,NAME_LEN);

	read_data(string(file_name),plhs[0],plhs[1]);
}
