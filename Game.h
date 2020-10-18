#include<iostream>
#include<vector>

using namespace std;

class Action
{
public:
	int m;
	int s0;
	vector<double> r;
	vector<vector<double> > t;
	vector<int> record;
	vector<vector<vector<double> > > transition_record;
	vector<vector<vector<double> > > power_record;
};

class Observation
{
public:
	int n;
	vector<int> s;
	vector<int> t;
	vector<vector<double> > real_distribution;
	vector<vector<double> > real_distribution_pseudo;
};

class State
{
public:
	int n;
	int m;
	vector<int> states;
	vector<int> times;
	vector<vector<double> > transitions;
	vector<double> rewards;
	vector<vector<State*> > Next_stat;
	bool avai;
	double V_value;
	int action;
};

class Direct_Trans_Observation
{
public:
	int action;
	int state;
	vector<int> transit_to;
	int num;
};

class Prior_Distribution
{
public:
	int n;
	int m;
	int size;
	vector<vector<vector<double> > > distribution;
};