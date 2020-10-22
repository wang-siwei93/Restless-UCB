#include<iostream>
#include"Game.h"
#include<random>
#include<fstream>
#include<cstring>
#include<time.h>
#include<algorithm>
#include<sys/time.h>
#include<math.h>

using namespace std;

int sumover(vector<int> v)
{
	int sum = 0;
	for(int i = 0; i < v.size(); i++)
		sum += v.at(i);
	return sum;
}

double maxin(vector<double> v)
{
	double max = v.at(0);
	for(int i = 0; i < v.size(); i++)
	{
		if(max < v.at(i))
			max = v.at(i);
	}
	return max;
}

double minin(double x, double y, double z)
{
	if(x < y)
		return x < z ? x : z;
	else
		return y < z ? y : z;
}

int maxindex(vector<double> v)
{
	double max = v.at(0);
	int index = 0;
	for(int i = 0; i < v.size(); i++)
	{
		if(max < v.at(i))
		{
			max = v.at(i);
			index = i;
		}
	}
	return index;
}

int minindex(vector<double> v)
{
	double min = v.at(0);
	int index = 0;
	for(int i = 0; i < v.size(); i++)
	{
		if(min > v.at(i))
		{
			min = v.at(i);
			index = i;
		}
	}
	return index;
}

double Uniform_Random();
int Bernoulli(double p);
int Random_State(vector<double> p);
void print_vector(vector<double> v);

vector<Action> Init_Actions(string filename);
vector<Action> Create_Actions(int N, int M, vector<int> S, vector<vector<double> > R, vector<vector<vector<double> > > T);

Observation Init(vector<Action> actions);
Observation Update(Observation ob, int arm, Action action);
int Next_State(int s, int t, Action action);
vector<vector<double> > Matrix_Power(vector<vector<double> > M, int t, Action action);
vector<vector<double> > Matrix_Multiply(vector<vector<double> > M, vector<vector<double> > N);
vector<vector<double> > Identical_Matrix(int n);
Observation Update2(Observation ob, int arm, vector<Action> actions);
vector<double> Next_distribution(vector<double> distribution, vector<vector<double> > transition);

double Game_default(vector<Action> actions, int T);
int Policy_default(Observation ob);

double Game_Whittle(vector<Action> actions, int T);
int Policy_Whittle(Observation ob, vector<Action> actions);
double Whittle_Index(vector<double> distribution, Action action);
int L_function(double omega1, double omega2, Action action);
double T_function(int t, double omega, Action action);

double Game_ReUCB(vector<Action> actions, int T);
double Omega_Star_function(double m, Action action, double start, double end);
double J_function(double m, Action action);
vector<vector<Direct_Trans_Observation> > Init_Direct_Trans_Observation(vector<Action> actions);
vector<Action> Estimate_UCB(vector<vector<Direct_Trans_Observation> > observations, vector<Action> actions, double rad);
int Policy_Whittle_Pseudo(Observation ob, vector<Action> actions);
Observation Update3(Observation ob, int arm, vector<Action> actions, vector<Action> actions2);

double Best_Average(vector<Action> actions);
vector<vector<vector<State> > > Available_States(vector<Action> actions, int TMAX);
vector<State> Avai_States(vector<Action> actions, int TMAX);
int next_con(State origin, int action, int trans, int TMAX);
State set_State(int i, vector<Action> actions, int TMAX);
double Solve_Bellman_M(vector<State> All_States);
double Solve_Bellman(vector<vector<vector<State> > > All_States);

double Game_TS(vector<Action> actions, int T, int size);
vector<double> Game_TS_vector(vector<Action> actions, int T, int size);
Prior_Distribution Init_Prior(vector<Action> actions, int size);
vector<Action> Sample_Actions(Prior_Distribution prior, vector<Action> actions);

double Game_UCRL(vector<Action> actions, int T);
double Game_UCRL_M(vector<Action> actions, int T);
vector<double> Game_UCRL_vector(vector<Action> actions, int T);
vector<double> Game_UCRL_vector_M(vector<Action> actions, int T);

double Game_TS_Pseudo(vector<Action> actions, int T);
double Game_TS_Pseudo2(vector<Action> actions, int T);
double Game_TS_Pseudo3(vector<Action> actions, int T);


vector<vector<double> > Regret_ReUCB(vector<Action> actions, int T, int number, double best);
vector<vector<double> > Regret_TS(vector<Action> actions, int T, int size, int number, double best);
vector<vector<double> > Regret_TS_vector(vector<Action> actions, int T, int size, int number, double best);
vector<vector<double> > Regret_UCRL(vector<Action> actions, int T, int number, double best);
vector<vector<double> > Regret_UCRL_vector(vector<Action> actions, int T, int number, double best);
vector<vector<double> > Regret_UCRL_vector_M(vector<Action> actions, int T, int number, double best);

int main()
{
	srand((unsigned)time(NULL));  


	vector<Action> actions = Init_Actions("Instances/Construct_Instance_1.txt");//Read the instance file


	double best_upper_bound = Best_Average(actions); //An upper bound for the average reward of the optimal policy
        double best_oracle = Game_Whittle(actions, 10000000)/10000000; //An average of the offline oracle
    
	vector<vector<double> > regrets_ReUCB = Regret_ReUCB(actions, 5000, 1000, best_oracle);//The Restless-UCB policy with T = 5000 to 50000 and 1000 runs
	print_vector(regrets_ReUCB[0]);//The average regrets
	print_vector(regrets_ReUCB[1]);//The variance of regrets
	
	vector<vector<double> > regrets_TS9 = Regret_TS_vector(actions, 5000, 9, 1000, best_oracle);//The TS policy with T = 5000 to 50000 and 1000 runs, its support on transition matrices are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]^2
	print_vector(regrets_TS9[0]);//The average regrets
	print_vector(regrets_TS9[1]);//The variance of regrets
	
	vector<vector<double> > regrets_TS4 = Regret_TS_vector(actions, 5000, 4, 1000, best_oracle);//The TS policy with T = 5000 to 50000 and 1000 runs, its support on transition matrices are [0.2, 0.4, 0.6, 0.8]^2
	print_vector(regrets_TS4[0]);//The average regrets
	print_vector(regrets_TS4[1]);//The variance of regrets
	
	vector<vector<double> > regrets_UCRL = Regret_UCRL_vector(actions, 5000, 1000, best_oracle);//The Colored-UCRL policy with T = 5000 to 50000 and 1000 runs
	print_vector(regrets_UCRL[0]);//The average regrets
	print_vector(regrets_UCRL[1]);//The variance of regrets
		
		
	//Only use the following commands for colored-UCRL2 when there are more than 2 arms
	//vector<vector<double> > regrets_UCRL = Regret_UCRL_vector_M(actions, 5000, 100, best_oracle); //The Colored-UCRL policy with T = 5000 to 50000 and 100 runs
	//print_vector(regrets_UCRL[0]);
        //print_vector(regrets_UCRL[1]);
	return 0;
}



double Uniform_Random()
{
	int max = 16384;
	double r = (rand() % max + 0.0)/max;
	return r;
}

int Bernoulli(double p)
{
	if(Uniform_Random() < p)
		return 1;
	return 0;
}

int Random_State(vector<double> p)
{
	double r = Uniform_Random();

	double q = 0;
	for(int i = 0; i < p.size(); i++)
	{
		q = q + p.at(i);
		if(q > r)
			return i;
	}
	return p.size()-1;

}

void print_vector(vector<double> v)
{
	for(int i = 0; i < v.size(); i++)
		cout << v.at(i) << endl;
	cout << endl;
}




vector<Action> Init_Actions(string filename)
{
	ifstream input(filename.c_str());

	int N;
	int M;

	input >> N;
	input >> M;

	vector<int> S(N);
	for(int i = 0; i < N; i++)
		input >> S.at(i);

	vector<vector<double> > R;
	for(int i = 0; i < N; i++)
	{
		vector<double> Ri(M);
		for(int j = 0; j < M; j++)
			input >> Ri.at(j);
		R.push_back(Ri);
	}

	vector<vector<vector<double> > > T;
	for(int i = 0; i < N; i++)
	{
		vector<vector<double> > Ti;
		for(int j = 0; j < M; j++)
		{
			vector<double> Tij(M);
			for(int k = 0; k < M; k++)
				input >> Tij.at(k);
			Ti.push_back(Tij);
		}
		T.push_back(Ti);
	}

	return Create_Actions(N, M, S, R, T);
}

vector<Action> Create_Actions(int N, int M, vector<int> S, vector<vector<double> > R, vector<vector<vector<double> > > T)
{
	vector<Action> actions(N);
	for(int i = 0; i < N; i++)
	{
		actions.at(i).m = M;
		actions.at(i).r = R.at(i);
		actions.at(i).s0 = S.at(i);
		actions.at(i).t = T.at(i);
		actions.at(i).power_record.push_back(T.at(i));
	}
	return actions;
}



Observation Init(vector<Action> actions)
{
	Observation ob;
	ob.n = actions.size();
	vector<int> states;
	vector<int> times;
	vector<vector<double> > real_distributions;
	for(int i = 0; i < ob.n; i++)
	{
		states.push_back(actions.at(i).s0);
		vector<double> distribution_i;
		for(int j = 0; j < actions.at(0).m; j++)
			distribution_i.push_back(actions.at(i).t.at(actions.at(i).s0).at(j));
		real_distributions.push_back(distribution_i);
		times.push_back(1);
	}
	ob.s = states;
	ob.t = times;
	ob.real_distribution = real_distributions;
	return ob;
}

Observation Update(Observation ob, int arm, Action action)
{
	Observation ob2;
	ob2.n = ob.n;
	vector<int> states;
	vector<int> times;
	for(int i = 0; i < ob.n; i++)
	{
		if(i != arm)
		{
			states.push_back(ob.s.at(i));
			times.push_back(ob.t.at(i) + 1);
		}
		else
		{
			states.push_back(Next_State(ob.s.at(i), ob.t.at(i), action));
			times.push_back(1);
		}
	}
	ob.s = states;
	ob.t = times;
	return ob;
}

int Next_State(int s, int t, Action action)
{
	if(t == 1)
		return Random_State(action.t.at(s));

	cout << "???" << endl;
	int n = action.record.size();
	int i = 0;
	for(i = 0; i < n; i++)
	{
		if(action.record.at(i) == t)
			break;
	}
	if(i < n)
		return Random_State(action.transition_record.at(i).at(s));
	vector<vector<double> > transition = Matrix_Power(action.t, t, action);

	action.record.push_back(t);
	action.transition_record.push_back(transition);
	return Random_State(transition.at(s));

}

vector<vector<double> > Matrix_Power(vector<vector<double> > M, int t, Action action)
{
	int n = action.power_record.size();
	vector<vector<double> > result = Identical_Matrix(M.size());
	int u = t;
	int v = 0;
	int index = 0;
	while(u > 0)
	{
		v = u % 2;
		u = u / 2;
		if(v == 1)
			result = Matrix_Multiply(result, action.power_record.at(index));
		index = index + 1;
		if(index >= n)
			action.power_record.push_back(Matrix_Multiply(action.power_record.back(), action.power_record.back()));
	}
	return result;
}

vector<vector<double> > Matrix_Multiply(vector<vector<double> > M, vector<vector<double> > N)
{
	vector<vector<double> > result;
	for(int i = 0; i < M.size(); i++)
	{
		vector<double> ri(M.size(), 0);
		for(int j = 0; j < M.size(); j++)
		{
			for(int k = 0; k < M.size(); k++)
				ri.at(j) += M.at(i).at(k) * N.at(k).at(j);
		}
		result.push_back(ri);
	}
	return result;
}

vector<vector<double> > Identical_Matrix(int n)
{
	vector<vector<double> > result;
	for(int i = 0; i < n; i++)
	{
		result.push_back(vector<double>(n,0));
		result.at(i).at(i) = 1;
	}
	return result;

}

Observation Update2(Observation ob, int arm, vector<Action> actions)
{
	Observation ob2;
	ob2.n = ob.n;
	vector<int> states;
	vector<vector<double> > real_distribution;
	vector<int> times;
	for(int i = 0; i < ob.n; i++)
	{
		vector<double> distribution_i;
		if(i != arm)
		{
			states.push_back(ob.s.at(i));
			times.push_back(ob.t.at(i) + 1);
			distribution_i = Next_distribution(ob.real_distribution.at(i), actions.at(i).t);
		}
		else
		{
			int state = Random_State(ob.real_distribution.at(i));
			states.push_back(state);
			times.push_back(1);
			distribution_i = actions.at(i).t.at(state);
		}
		real_distribution.push_back(distribution_i);
	}
	ob.s = states;
	ob.t = times;
	ob.real_distribution = real_distribution;
	return ob;
}

vector<double> Next_distribution(vector<double> distribution, vector<vector<double> > transition)
{
	vector<double> d;
	int n = distribution.size();
	for(int i = 0; i < n; i++)
	{
		double sum = 0;
		for(int j = 0; j < n; j++)
			sum += distribution.at(j) * transition.at(j).at(i);
		d.push_back(sum);
	}
	return d;
}



double Game_default(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	while(t < T)
	{
		int arm = Policy_default(ob);
		ob = Update2(ob,arm,actions);
		reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
		t = t + 1;
	}
	return reward;
}

int Policy_default(Observation ob)
{
	return rand()%ob.n;
}



double Game_Whittle(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	while(t < T)
	{
		int arm = Policy_Whittle(ob, actions);
		ob = Update2(ob,arm,actions);
		reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
		t = t + 1;
	}
	return reward;
}

int Policy_Whittle(Observation ob, vector<Action> actions)
{
	vector<double> index;
	for(int i = 0; i < ob.n; i++)
		index.push_back(Whittle_Index(ob.real_distribution.at(i), actions.at(i)));
	double max = -1000;
	int j = -1;
	for(int i = 0; i < ob.n; i++)
	{
		if(max < index.at(i))
		{
			max = index.at(i);
			j = i;
		}
	}
	return j;
}

double Whittle_Index(vector<double> distribution, Action action)
{
	if(action.t.at(1).at(1) >= action.t.at(0).at(1))
	{
		double omega = distribution.at(1);
		double omega0 = action.t.at(0).at(1)/(action.t.at(0).at(1) + action.t.at(1).at(0));
		if(omega <= action.t.at(0).at(1))
			return omega * (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else if(omega >= action.t.at(1).at(1))
			return omega * (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else if(omega >= omega0)
			return omega/(1-action.t.at(1).at(1)+omega)* (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else
		{
			int L = L_function(action.t.at(0).at(1), omega, action);
			double p1 = T_function(1,omega,action);
			double p2 = T_function(L,action.t.at(0).at(1), action);
			return ((omega - p1)*(L+1) + p2)/(1 - action.t.at(1).at(1) + (omega - p1)*L + p2) * (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		}
	}
	else
	{
		double omega = distribution.at(1);
		double omega0 = action.t.at(0).at(1)/(action.t.at(0).at(1) + action.t.at(1).at(0));
		double t = T_function(1,action.t.at(1).at(1),action);
		double p1 = T_function(1,omega,action);
		if(omega <= action.t.at(1).at(1))
			return omega * (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else if(omega >= action.t.at(0).at(1))
			return omega * (action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else if(omega < omega0)
			return (omega+action.t.at(0).at(1)-p1)/(1+action.t.at(0).at(1)-t+p1-omega)*(action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else if(omega < t)
			return action.t.at(0).at(1)/(1+action.t.at(0).at(1)-t)*(action.r.at(1) - action.r.at(0)) + action.r.at(0);
		else
			return action.t.at(0).at(1)/(1+action.t.at(0).at(1)-omega)*(action.r.at(1) - action.r.at(0)) + action.r.at(0);
	}
		
		
		
		
		
		
		
	//else if(omega >= omega0)
	//	return omega/(1-action.t.at(1).at(1)+omega)* action.r.at(1);
	//else
	//{
	//	int L = L_function(action.t.at(0).at(1), omega, action);
	//	double p1 = T_function(1,omega,action);
	//	double p2 = T_function(L,action.t.at(0).at(1), action);
	//	return ((omega - p1)*(p2 + (L)*(1-action.t.at(1).at(1))) + p2)/(1 - action.t.at(1).at(1) + p2) * action.r.at(1);
	//}

}

int L_function(double omega1, double omega2, Action action)
{
	double p1 = action.t.at(1).at(1) - action.t.at(0).at(1);
	double p2 = log((action.t.at(0).at(1) - omega2*(1-p1))/(action.t.at(0).at(1) - omega1*(1-p1)));
	double p3 = log(p1);
	return int(p2/p3)+1 ;
}

double T_function(int t, double omega, Action action)
{
	vector<vector<double> > transition = Matrix_Power(action.t, t, action);
	return (1-omega)*transition.at(0).at(1) + omega * transition.at(1).at(1);
}



double Game_ReUCB(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	vector<vector<Direct_Trans_Observation> > observations = Init_Direct_Trans_Observation(actions);
	int m_t = (int) pow(T, 2.0/3) + 1;
	int start = 0;
	int n = actions.size();
	int m = actions.at(0).m;
	while(start < n)
	{
		int arm = start;
		int finish = 0;
		while(finish < m)
		{
			int start_state = ob.s.at(start);
			ob = Update2(ob,arm,actions);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			int end_state = ob.s.at(start);
			observations.at(start).at(start_state).num += 1;
			observations.at(start).at(start_state).transit_to.at(end_state) += 1;
			
			if(observations.at(start).at(start_state).num == m_t)
				finish += 1;
			t = t + 1;
		}
		start += 1;
	}

	double rad = sqrt(log(T)/2/m_t);

	vector<Action> actions_2 = Estimate_UCB(observations, actions, rad);
	
	vector<vector<double> > pseudo_distributions;
	for(int i = 0; i < ob.n; i++)
	{
		vector<double> start(actions.at(0).m, 0.0);
		start.at(ob.s.at(i)) = 1.0;
		pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
	}
	ob.real_distribution_pseudo = pseudo_distributions;

	while(t < T)
	{
		int arm = Policy_Whittle_Pseudo(ob, actions_2);
		ob = Update3(ob,arm,actions, actions_2);
		reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
		t = t + 1;
	}
	return reward;
}

double Omega_Star_function(double m, Action action, double start, double end)
{
	double mid = (end + start)/2;
	if(end - start < 0.00001)
		return mid;

	vector<double> d;
	d.push_back(1-mid);
	d.push_back(mid);

	double v = Whittle_Index(d, action);

	if(v < m)
		return Omega_Star_function(m, action, mid, end);
	else
		return Omega_Star_function(m, action, start, mid);
}

double J_function(double m, Action action)
{
	double omega_0 = action.t.at(0).at(1)/(action.t.at(0).at(1) + action.t.at(1).at(0));
	double omega_star = Omega_Star_function(m,action, 0.0, 1.0);
	if(omega_star < action.t.at(0).at(1))
		return omega_0 * (action.r.at(1)-action.r.at(0)) + action.r.at(0);
	else if(omega_star >= omega_0)
		return m;
	else
	{
		int L = L_function(action.t.at(0).at(1), omega_star, action);
		double p1 = T_function(L,action.t.at(0).at(1), action);
		return ((1-action.t.at(1).at(1)) * m * (L) + p1 * action.r.at(1) + (1-action.t.at(1).at(1)) * action.r.at(0))/((1-action.t.at(1).at(1)) * (L+1) + p1);
	}

}

vector<vector<Direct_Trans_Observation> > Init_Direct_Trans_Observation(vector<Action> actions)
{
	vector<vector<Direct_Trans_Observation> > observations;
	int n = actions.size();
	int m = actions.at(0).m;
	for(int i = 0; i < n; i++)
	{
		vector<Direct_Trans_Observation> ob_i;
		for(int j = 0; j < m; j++)
		{
			Direct_Trans_Observation ob_ij;
			ob_ij.action = i;
			ob_ij.state = j;
			vector<int> trans_to(m,0);
			ob_ij.transit_to = trans_to;
			ob_ij.num = 0;
			ob_i.push_back(ob_ij);
		}
		observations.push_back(ob_i);
	}
	return observations;

}

vector<Action> Estimate_UCB(vector<vector<Direct_Trans_Observation> > observations, vector<Action> actions, double rad)
{
	int N = observations.size();;
	int M = observations.at(0).size();

	vector<int> S(N,0);

	vector<vector<double> > R;
	for(int i = 0; i < N; i++)
	{
		vector<double> Ri(M);
		for(int j = 0; j < M; j++)
			Ri.at(j) = actions.at(i).r.at(j);
		R.push_back(Ri);
	}

	vector<vector<vector<double> > > T;
	for(int i = 0; i < N; i++)
	{
		vector<vector<double> > Ti;
		for(int j = 0; j < M; j++)
		{
			vector<double> Tij(M);
			for(int k = 0; k < M; k++)
			{
				if(k == 0)
					Tij.at(k) = (observations.at(i).at(j).transit_to.at(k)+0.0)/observations.at(i).at(j).num - rad;
				if(k == M)
					Tij.at(k) = (observations.at(i).at(j).transit_to.at(k)+0.0)/observations.at(i).at(j).num + rad;
				else
					Tij.at(k) = (observations.at(i).at(j).transit_to.at(k)+0.0)/observations.at(i).at(j).num;
			}
			Ti.push_back(Tij);
		}
		T.push_back(Ti);
	}

	return Create_Actions(N, M, S, R, T);
}

int Policy_Whittle_Pseudo(Observation ob, vector<Action> actions)
{
	vector<double> index;
	for(int i = 0; i < ob.n; i++)
		index.push_back(Whittle_Index(ob.real_distribution_pseudo.at(i), actions.at(i)));
	double max = -1000;
	int j = -1;
	for(int i = 0; i < ob.n; i++)
	{
		if(max < index.at(i))
		{
			max = index.at(i);
			j = i;
		}
	}
	//cout << index.at(0) << index.at(1) << endl;
	return j;
}

Observation Update3(Observation ob, int arm, vector<Action> actions, vector<Action> actions2)
{
	if(ob.real_distribution_pseudo.size() == 0)
	{
		vector<vector<double> > pseudo_distributions;
		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions2.at(i).t, ob.t.at(i), actions2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
	}
	Observation ob2;
	ob2.n = ob.n;
	vector<int> states;
	vector<vector<double> > real_distribution;
	vector<vector<double> > pseudo_distribution;
	vector<int> times;
	for(int i = 0; i < ob.n; i++)
	{
		vector<double> distribution_i;
		vector<double> pseudo_distribution_i;
		if(i != arm)
		{
			states.push_back(ob.s.at(i));
			times.push_back(ob.t.at(i) + 1);
			distribution_i = Next_distribution(ob.real_distribution.at(i), actions.at(i).t);
			if(ob.real_distribution_pseudo.size() == 0)
			{
				vector<double> start(actions2.at(0).m, 0.0);
				start.at(ob.s.at(i)) = 1.0;
				pseudo_distribution_i = Next_distribution(start, Matrix_Power(actions2.at(i).t, ob.t.at(i), actions2.at(i)));
			}
			else
				pseudo_distribution_i = Next_distribution(ob.real_distribution_pseudo.at(i), actions2.at(i).t);
		}
		else
		{
			int state = Random_State(ob.real_distribution.at(i));
			states.push_back(state);
			times.push_back(1);
			distribution_i = actions.at(i).t.at(state);
			pseudo_distribution_i = actions2.at(i).t.at(state);
		}
		real_distribution.push_back(distribution_i);
		pseudo_distribution.push_back(pseudo_distribution_i);
	}
	ob.s = states;
	ob.t = times;
	ob.real_distribution = real_distribution;
	ob.real_distribution_pseudo = pseudo_distribution;
	return ob;
}



double Best_Average(vector<Action> actions)
{
	double min = 1000;
	double delta = 0.001;
	double start = 0;
	double end = 1;
	vector<double> q(actions.size(), 0.0);
	while(start <= end)
	{
		//if(start > 0.14)
		//	cout << endl;
		double sum = 0 - start*(actions.size()-1);
		for(int i = 0; i < actions.size(); i++)
		{
			double qi = J_function(start, actions.at(i));
			q.at(i) = qi;
			//q.at(i) = qi > q.at(i) ? qi : q.at(i);
			sum += q.at(i);
			//sum += J_function(start, actions.at(i));
		}
		//cout << start <<" "<<J_function(start, actions.at(0)) << endl;
		//cout << start <<" "<< Omega_Star_function(start, actions.at(1), 0.0, 1.0) << endl;
		if(min > sum)
			min = sum;
		start += delta;
	}
	return min;

}

vector<State> Avai_States(vector<Action> actions, int TMAX)
{
	int max_states = 1;
	for(int i = 0; i < actions.size(); i++)
		max_states *= actions.at(0).m * TMAX;
	vector<State> avai_States;
	for(int i = 0; i < max_states; i++)
	{
		State new_state = set_State(i, actions, TMAX);
		avai_States.push_back(new_state);
	}
	for(int i = 0; i < max_states; i++)
	{
		vector<vector<State*> > next_state;
		for(int j = 0; j < actions.size(); j++)
		{
			vector<State* > next_state2;
			for(int k = 0 ; k < actions.at(0).m; k++)
				next_state2.push_back(& avai_States.at(next_con(avai_States.at(i), j, k, TMAX)));
			next_state.push_back(next_state2);
		}
		avai_States.at(i).Next_stat = next_state;
	}
	return avai_States;
}


double Solve_Bellman_M(vector<State> All_States)
{
	double gap = 10000;
	int NOS = All_States.size();
	int n = All_States.at(0).n;
	int m = All_States.at(0).m;
	double max,min;
	int iter = 1;
	while(gap > 0.001)
	{
		vector<State> All_States2(All_States);
		max = -10000;
		min = 10000;
		for(int i = 0; i < NOS; i++)
		{
			double V = -10000;
			for(int k = 0; k < n; k++)
			{
				double sum = All_States.at(i).rewards.at(k);
				double sum2 = 0;
				for(int k2 = 0; k2 < m; k2++)
					sum2 += All_States.at(i).transitions.at(k).at(k2) * All_States.at(i).Next_stat.at(k).at(k2)->V_value;
				sum += sum2;
				if(sum > V)
					V = sum;
			}
			All_States2.at(i).V_value = V;
		}
		for(int i = 0; i < NOS; i++)
		{
			double Vprime = All_States.at(i).V_value;
			double V = All_States2.at(i).V_value;
			All_States.at(i).V_value = V;
			double gap2 = Vprime - V;
			if(gap2 > max)
				max = gap2;
			if(gap2 < min)
				min = gap2;
		}
		gap = max - min;
		iter += 1;
	}
	return -(max+min)/2;
}

int next_con(State origin, int action, int trans, int TMAX)
{
	vector<int> ids;
	for(int i = 0; i < origin.n; i++)
	{
		int id = origin.states.at(i) * TMAX + (origin.times.at(i) + 1 > TMAX ? TMAX - 1 : origin.times.at(i));
		ids.push_back(id);
	}
	ids.at(action) = trans * TMAX;
	int number = 0;
	for(int j = origin.n - 1; j >= 0; j--)
	{
		number *= TMAX * origin.m;
		number += ids.at(j);
	}
	return number;

}

State set_State(int i, vector<Action> actions, int TMAX)
{
	State new_state;
	vector<int> ids;
	int number = i;
	for(int j = 0; j < actions.size(); j++)
	{
		int id = number % (actions.at(0).m * TMAX);
		ids.push_back(id);
		number = number / (actions.at(0).m * TMAX);
	}
	new_state.n = actions.size();
	new_state.m = actions.at(0).m;
	vector<int> new_states;
	vector<int> new_times;
	for(int j = 0; j < actions.size(); j++)
	{
		new_states.push_back(ids.at(j) / TMAX);
		new_times.push_back(ids.at(j) % TMAX + 1);
	}
	new_state.states = new_states;
	new_state.times = new_times;
	new_state.V_value = 0;
	vector<vector<double> > new_transition;
	vector<double> new_rewards;
	for(int j = 0; j < new_state.n; j++)
	{
		vector<double> s_j;
		for(int k = 0; k < new_state.m; k++)
		{
			if(k == new_states.at(j))
				s_j.push_back(1.0);
			else
				s_j.push_back(0.0);
		}
		vector<double> tran_j = Next_distribution(s_j, Matrix_Power(actions.at(j).t, new_times.at(j), actions.at(j)));
		new_transition.push_back(tran_j);
		double sum = 0;
		for(int k = 0; k < new_state.m; k++)
			sum += tran_j.at(k) * actions.at(j).r.at(k);
		new_rewards.push_back(sum);
	}
	new_state.transitions = new_transition;
	new_state.rewards = new_rewards;
	return new_state;
}

double Game_UCRL_M(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;

	double delta = 0.001;

	int TMAX = 8;
	int n = actions.size();
	int m = actions.at(0).m;
	int NOS = 1;
	for(int i = 0; i < n; i++)
		NOS *= (m * TMAX);
	vector<vector<vector<vector<int> > > > non_trans_observation;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}
	int t_k = 0;
	while(t < T)
	{
		bool stop = false;
		vector<State > All_States = Avai_States(actions,TMAX);
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		double gap = 10000;
		double max,min;
		int iter = 0;
		while(gap > 0.001 && iter <= 50)
		{
			vector<State > All_States2(All_States);
			max = -10000;
			min = 10000;
			for(int i = 0; i < NOS; i++)
			{
				double V = -10000;
				for(int k = 0; k < n; k++)
				{
					double sum = 0;
					vector<double> V_values(m,0);
					for(int j = 0; j < m; j++)
					{
						V_values.at(j) = All_States.at(i).Next_stat.at(k).at(j)->V_value + actions.at(k).r.at(j);
					}
					int start_state = All_States.at(i).states.at(k);
					int start_time = All_States.at(i).times.at(k);
					vector<int > numbers = non_trans_observation.at(k).at(start_state).at(start_time-1);
					int number = 0;
					for(int j = 0; j < m; j++)
						number += numbers.at(j);
					if(number == 0)
					{
						sum = maxin(V_values);
						if(sum < -10)
							cout << "?" << endl;
					}
					else
					{
						int maximum = maxindex(V_values);
						int minimum = minindex(V_values);

						double max_p = (numbers.at(maximum)+0.0)/number;
						double min_p = (numbers.at(minimum)+0.0)/number;
						double rad = sqrt(56*2*log(4*t_k/delta)/2/number);

						double minimum_probability = minin(max_p, min_p, rad);
						for(int j = 0; j < m; j++)
							sum += (numbers.at(j)+0.0)/number * V_values.at(j);
						sum += minimum_probability * (V_values.at(maximum) - V_values.at(minimum));
						if(sum < -10)
							cout << "?" << endl;
					}
					if(sum > V)
					{
						All_States.at(i).action = k;
						V = sum;
					}
					if(V < -10)
						cout << "?" << endl;
				}
				All_States2.at(i).V_value = V;
			}
			for(int i = 0; i < NOS; i++)
			{
				double Vprime = All_States.at(i).V_value;
				double V = All_States2.at(i).V_value;
				All_States.at(i).V_value = V;
				double gap2 = Vprime - V;
				if(gap2 > max)
					max = gap2;
				if(gap2 < min)
					min = gap2;
			}
			gap = max - min;
			iter = iter + 1;	
		}
		//cout << (max + min)/2 << endl;
		//cout << max << " " << min << endl;
		while((!stop) && t < T)
		{
			vector<int> ids;
			for(int i = 0; i < n; i++)
			{
				int id = ob.s.at(i) * TMAX + (ob.t.at(i) - 1 > TMAX - 1 ? TMAX - 1 : ob.t.at(i) - 1);
				ids.push_back(id);
			}
			int number = 0;
			for(int j = n - 1; j >= 0; j--)
			{
				number *= TMAX * m;
				number += ids.at(j);
			}
			State actionsss = All_States.at(number);
			int arm = actionsss.action;
			int start_state = actionsss.states.at(arm);
			int start_time = actionsss.times.at(arm);
			ob = Update2(ob,arm,actions);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;

			non_trans_observation.at(arm).at(start_state).at(start_time-1).at(end_state) += 1;
			int u = sumover(non_trans_observation.at(arm).at(start_state).at(start_time-1));
			int v = sumover(non_trans_observation2.at(arm).at(start_state).at(start_time-1));
			if(u > v * 2)
				stop = true;
		}
		t_k = t;
		//cout << t << endl;
	}
	return reward;
}


vector<double> Game_UCRL_vector_M(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;

	double delta = 0.001;
	vector<double> rewards(1,0.0);

	int TMAX = 3;
	int n = actions.size();
	int m = actions.at(0).m;
	int NOS = 1;
	for(int i = 0; i < n; i++)
		NOS *= (m * TMAX);
	vector<vector<vector<vector<int> > > > non_trans_observation;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}
	int t_k = 0;
	while(t < 10*T)
	{
		bool stop = false;
		vector<State > All_States = Avai_States(actions,TMAX);
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		double gap = 10000;
		double max,min;
		int iter = 0;
		while(gap > 0.001 && iter <= 100)
		{
			vector<State > All_States2(All_States);
			max = -10000;
			min = 10000;
			for(int i = 0; i < NOS; i++)
			{
				double V = -10000;
				for(int k = 0; k < n; k++)
				{
					double sum = 0;
					vector<double> V_values(m,0);
					for(int j = 0; j < m; j++)
					{
						V_values.at(j) = All_States.at(i).Next_stat.at(k).at(j)->V_value + actions.at(k).r.at(j);
					}
					int start_state = All_States.at(i).states.at(k);
					int start_time = All_States.at(i).times.at(k);
					vector<int > numbers = non_trans_observation.at(k).at(start_state).at(start_time-1);
					int number = 0;
					for(int j = 0; j < m; j++)
						number += numbers.at(j);
					if(number == 0)
					{
						sum = maxin(V_values);
						if(sum < -10)
							cout << "?" << endl;
					}
					else
					{
						int maximum = maxindex(V_values);
						int minimum = minindex(V_values);

						double max_p = (numbers.at(maximum)+0.0)/number;
						double min_p = (numbers.at(minimum)+0.0)/number;
						double rad = sqrt(56*2*log(4*t_k/delta)/2/number);

						double minimum_probability = minin(max_p, min_p, rad);
						for(int j = 0; j < m; j++)
							sum += (numbers.at(j)+0.0)/number * V_values.at(j);
						sum += minimum_probability * (V_values.at(maximum) - V_values.at(minimum));
						if(sum < -10)
							cout << "?" << endl;
					}
					if(sum > V)
					{
						All_States.at(i).action = k;
						V = sum;
					}
					if(V < -10)
						cout << "?" << endl;
				}
				All_States2.at(i).V_value = V;
			}
			for(int i = 0; i < NOS; i++)
			{
				double Vprime = All_States.at(i).V_value;
				double V = All_States2.at(i).V_value;
				All_States.at(i).V_value = V;
				double gap2 = Vprime - V;
				if(gap2 > max)
					max = gap2;
				if(gap2 < min)
					min = gap2;
			}
			gap = max - min;
			iter = iter + 1;	
		}
		//cout << (max + min)/2 << endl;
		//cout << max << " " << min << endl;
		while((!stop) && t < 10*T)
		{
			vector<int> ids;
			for(int i = 0; i < n; i++)
			{
				int id = ob.s.at(i) * TMAX + (ob.t.at(i) - 1 > TMAX - 1 ? TMAX - 1 : ob.t.at(i) - 1);
				ids.push_back(id);
			}
			int number = 0;
			for(int j = n - 1; j >= 0; j--)
			{
				number *= TMAX * m;
				number += ids.at(j);
			}
			State actionsss = All_States.at(number);
			int arm = actionsss.action;
			int start_state = actionsss.states.at(arm);
			int start_time = actionsss.times.at(arm);
			ob = Update2(ob,arm,actions);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;
			if(t % T == 0)
				rewards.push_back(reward);

			non_trans_observation.at(arm).at(start_state).at(start_time-1).at(end_state) += 1;
			int u = sumover(non_trans_observation.at(arm).at(start_state).at(start_time-1));
			int v = sumover(non_trans_observation2.at(arm).at(start_state).at(start_time-1));
			if(u > v * 2)
				stop = true;
		}
		t_k = t;
		//cout << t << endl;
	}
	return rewards;
}




vector<vector<vector<State> > > Available_States(vector<Action> actions, int TMAX)
{
	vector<vector<vector<State> > > avai;
	if(actions.size() > 2)
		return avai;
	int m = actions.at(0).m;
	int n = actions.size();
	vector<vector<State> > avai_i;
	for(int i = 0; i < m; i++)
	{
		vector<State> avai_ij;
		for(int t = 0; t < TMAX; t++)
		{
			State a;

			a.avai = false;
			a.V_value = 0.0;

			vector<int> states;
			states.push_back(0);
			states.push_back(i);
			a.states = states;

			vector<int> times;
			times.push_back(1);
			times.push_back(t+2);
			a.times = times;

			vector<vector<double> > transition;
			vector<double> rewards;
			for(int j = 0; j < n; j++)
			{
				vector<double> s_j;
				for(int k = 0; k < m; k++)
				{
					if(k == states.at(j))
						s_j.push_back(1.0);
					else
						s_j.push_back(0.0);
				}
				vector<double> tran_j = Next_distribution(s_j, Matrix_Power(actions.at(j).t, times.at(j), actions.at(j)));
				transition.push_back(tran_j);
				double sum = 0;
				for(int k = 0; k < m; k++)
					sum += tran_j.at(k) * actions.at(j).r.at(k);
				rewards.push_back(sum);
			}
			a.transitions = transition;
			a.rewards = rewards;
			avai_ij.push_back(a);
		}
		avai_i.push_back(avai_ij);
	}
	avai.push_back(avai_i);

	vector<vector<State> > avai_i2;
	for(int i = 0; i < m; i++)
	{
		vector<State> avai_ij2;
		for(int t = 0; t < TMAX; t++)
		{
			State a;

			a.avai = false;
			a.V_value = 0.0;

			vector<int> states;
			states.push_back(1);
			states.push_back(i);
			a.states = states;

			vector<int> times;
			times.push_back(1);
			times.push_back(t+2);
			a.times = times;

			vector<vector<double> > transition;
			vector<double> rewards;
			for(int j = 0; j < n; j++)
			{
				vector<double> s_j;
				for(int k = 0; k < m; k++)
				{
					if(k == states.at(j))
						s_j.push_back(1.0);
					else
						s_j.push_back(0.0);
				}
				vector<double> tran_j = Next_distribution(s_j, Matrix_Power(actions.at(j).t, times.at(j), actions.at(j)));
				transition.push_back(tran_j);
				double sum = 0;
				for(int k = 0; k < m; k++)
					sum += tran_j.at(k) * actions.at(j).r.at(k);
				rewards.push_back(sum);
			}
			a.transitions = transition;
			a.rewards = rewards;
			avai_ij2.push_back(a);
		}
		avai_i2.push_back(avai_ij2);
	}
	avai.push_back(avai_i2);

	vector<vector<State> > avai_i3;
	for(int i = 0; i < m; i++)
	{
		vector<State> avai_ij3;
		for(int t = 0; t < TMAX; t++)
		{
			State a;

			a.avai = false;
			a.V_value = 0.0;

			vector<int> states;
			states.push_back(i);
			states.push_back(0);
			a.states = states;

			vector<int> times;
			times.push_back(t+2);
			times.push_back(1);
			a.times = times;

			vector<vector<double> > transition;
			vector<double> rewards;
			for(int j = 0; j < n; j++)
			{
				vector<double> s_j;
				for(int k = 0; k < m; k++)
				{
					if(k == states.at(j))
						s_j.push_back(1.0);
					else
						s_j.push_back(0.0);
				}
				vector<double> tran_j = Next_distribution(s_j, Matrix_Power(actions.at(j).t, times.at(j), actions.at(j)));
				transition.push_back(tran_j);
				double sum = 0;
				for(int k = 0; k < m; k++)
					sum += tran_j.at(k) * actions.at(j).r.at(k);
				rewards.push_back(sum);
			}
			a.transitions = transition;
			a.rewards = rewards;
			avai_ij3.push_back(a);
		}
		avai_i3.push_back(avai_ij3);
	}
	avai.push_back(avai_i3);

	vector<vector<State> > avai_i4;
	for(int i = 0; i < m; i++)
	{
		vector<State> avai_ij4;
		for(int t = 0; t < TMAX; t++)
		{
			State a;

			a.avai = false;
			a.V_value = 0.0;

			vector<int> states;
			states.push_back(i);
			states.push_back(1);
			a.states = states;

			vector<int> times;
			times.push_back(t+2);
			times.push_back(1);
			a.times = times;

			vector<vector<double> > transition;
			vector<double> rewards;
			for(int j = 0; j < n; j++)
			{
				vector<double> s_j;
				for(int k = 0; k < m; k++)
				{
					if(k == states.at(j))
						s_j.push_back(1.0);
					else
						s_j.push_back(0.0);
				}
				vector<double> tran_j = Next_distribution(s_j, Matrix_Power(actions.at(j).t, times.at(j), actions.at(j)));
				transition.push_back(tran_j);
				double sum = 0;
				for(int k = 0; k < m; k++)
					sum += tran_j.at(k) * actions.at(j).r.at(k);
				rewards.push_back(sum);
			}
			a.transitions = transition;
			a.rewards = rewards;
			avai_ij4.push_back(a);
		}
		avai_i4.push_back(avai_ij4);
	}
	avai.push_back(avai_i4);

	for(int l = 0; l < 4; l++)
	{
		for(int i = 0; i < m; i++)
		{
			for(int t = 0; t < TMAX; t++)
			{
				vector<vector<State*> > next;
				for(int j = 0; j < n; j++)
				{
					vector<State*> next_j;
					int T = avai.at(l).at(i).at(t).times.at(1-j)-2;
					int S = avai.at(l).at(i).at(t).states.at(1-j);
					T = T < TMAX-1 ? T+1 : T;
					for(int k = 0; k < m; k++)
						next_j.push_back(&avai.at(j*2+k).at(S).at(T));
					next.push_back(next_j);
				}
				avai.at(l).at(i).at(t).Next_stat = next;
			}
		}
	}
	return avai;
}

double Solve_Bellman(vector<vector<vector<State> > > All_States)
{
	double gap = 10000;
	int u = All_States.size();
	int v = All_States.at(0).size();
	int w = All_States.at(0).at(0).size();
	int n = 2;
	double max,min;
	int iter = 1;
	while(gap > 0.001)
	{
		vector<vector<vector<State> > > All_States2(All_States);
		max = -10000;
		min = 10000;
		for(int l = 0; l < u; l++)
		{
			for(int i = 0; i < v; i++)
			{
				for(int t = 0; t < w; t++)
				{
					double V = -10000;
					for(int k = 0; k < 2; k++)
					{
						double sum = All_States.at(l).at(i).at(t).rewards.at(k);
						double sum2 = 0;
						for(int k2 = 0; k2 < v; k2++)
							sum2 += All_States.at(l).at(i).at(t).transitions.at(k).at(k2) * All_States.at(l).at(i).at(t).Next_stat.at(k).at(k2)->V_value;
						sum += sum2;
						//sum = (sum + sum2*iter)/(iter + 1);
						if(sum > V)
							V = sum;
					}
					All_States2.at(l).at(i).at(t).V_value = V;
				}
			}
		}
		for(int l = 0; l < u; l++)
		{
			for(int i = 0; i < v; i++)
			{
				for(int t = 0; t < w; t++)
				{
					double Vprime = All_States.at(l).at(i).at(t).V_value;
					double V = All_States2.at(l).at(i).at(t).V_value;
					All_States.at(l).at(i).at(t).V_value = V;
					//cout << V << endl;
					//cout << Vprime << endl;
					double gap2 = Vprime - V;
					if(gap2 > max)
						max = gap2;
					if(gap2 < min)
						min = gap2;
				}
			}
		}
		gap = max - min;
		iter += 1;
	}
	return -(max+min)/2;
}



double Game_TS(vector<Action> actions, int T, int size)
{
	double check_max = -100;
	double MAX_GAP = 100;
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	Prior_Distribution prior = Init_Prior(actions, size);
	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}

	vector<vector<vector<vector<vector<double> > > > > probabilities;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<vector<double> > > > pro_i;
		for(int j = 0; j < size * size; j++)
		{
			vector<vector<vector<double> > > pro_ij;
			for(int k = 0; k < m; k++)
			{
				vector<vector<double> > pro_ijk;
				for(int l = 0; l < TMAX; l++)
				{
					vector<double> pro_ijkl;
					if(l == 0)
					{
						pro_ijkl.push_back(1 - prior.distribution.at(i).at(j).at(k+1));
						pro_ijkl.push_back(prior.distribution.at(i).at(j).at(k+1));
					}
					else
					{
						double p1 = pro_ijk.at(l-1).at(0) * prior.distribution.at(i).at(j).at(1) + pro_ijk.at(l-1).at(1) * prior.distribution.at(i).at(j).at(2);
						pro_ijkl.push_back(1 - p1);
						pro_ijkl.push_back(p1);
					}
					pro_ijk.push_back(pro_ijkl);
				}
				pro_ij.push_back(pro_ijk);
			}
			pro_i.push_back(pro_ij);
		}
		probabilities.push_back(pro_i);
	}



	while(t < T)
	{

		bool stop = false;
		vector<Action> actions_2 = Sample_Actions(prior, actions);
		vector<vector<double> > pseudo_distributions;

		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		while((!stop) && t < T)
		{
			int arm = Policy_Whittle_Pseudo(ob, actions_2);
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update3(ob,arm,actions, actions_2);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;

			non_trans_observation.at(arm).at(start_state).at(running_time).at(end_state) += 1;
			int u = non_trans_observation.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation.at(arm).at(start_state).at(running_time).at(1);
			int v = non_trans_observation2.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation2.at(arm).at(start_state).at(running_time).at(1);
			if(u > v * 2)
				stop = true;
		}

		for(int i = 0; i < n; i++)
		{
			double sum_pro = 0;
			vector<double> sums;
			double min = 100000000;
			double max = -100000000;
			for(int j = 0; j < size * size; j++)
			{
				if(prior.distribution.at(i).at(j).at(0) == 0)
				{
					//cout << j << endl;
					sums.push_back(0);
				}
				else
				{
					double sum = 0;
					for(int k = 0; k < m; k++)
					{
						for(int l = 0; l < TMAX; l++)
						{
							for(int l2 = 0; l2 < m; l2++)
							{
								int numbers = non_trans_observation.at(i).at(k).at(l).at(l2) - non_trans_observation2.at(i).at(k).at(l).at(l2);
								sum += numbers * log(probabilities.at(i).at(j).at(k).at(l).at(l2));
							}
						}
					}
					if(sum < min)
						min = sum;
					if(sum > max)
						max = sum;
					sums.push_back(sum);
				}
			}
			//cout << "maxmin" << max << endl << min << endl;
			for(int j = 0; j < size * size; j++)
			{
				double uvw = max - sums.at(j);
				//cout << sums.at(j) << endl;
				//if(uvw > check_max)
				//{
				//	check_max = uvw;
				//	cout << check_max << endl;
				//}
				if(uvw >= MAX_GAP)
					prior.distribution.at(i).at(j).at(0) = 0;
				else if(prior.distribution.at(i).at(j).at(0) != 0)
					prior.distribution.at(i).at(j).at(0) *= exp(MAX_GAP - uvw);
				//cout << prior.distribution.at(i).at(j).at(0) << endl;
				sum_pro = sum_pro + prior.distribution.at(i).at(j).at(0);
			}
			//cout << endl;
			for(int j = 0; j < size * size; j++)
			{
				prior.distribution.at(i).at(j).at(0) /= sum_pro;
				//cout << prior.distribution.at(i).at(j).at(0) << endl;
			}
			//cout << endl;
		}
		//cout << t << endl;

	}
	return reward;
}

vector<double> Game_TS_vector(vector<Action> actions, int T, int size)
{
	double check_max = -100;
	vector<double> rewards(1,0.0);
	double MAX_GAP = 100;
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	Prior_Distribution prior = Init_Prior(actions, size);
	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}

	vector<vector<vector<vector<vector<double> > > > > probabilities;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<vector<double> > > > pro_i;
		for(int j = 0; j < size * size; j++)
		{
			vector<vector<vector<double> > > pro_ij;
			for(int k = 0; k < m; k++)
			{
				vector<vector<double> > pro_ijk;
				for(int l = 0; l < TMAX; l++)
				{
					vector<double> pro_ijkl;
					if(l == 0)
					{
						pro_ijkl.push_back(1 - prior.distribution.at(i).at(j).at(k+1));
						pro_ijkl.push_back(prior.distribution.at(i).at(j).at(k+1));
					}
					else
					{
						double p1 = pro_ijk.at(l-1).at(0) * prior.distribution.at(i).at(j).at(1) + pro_ijk.at(l-1).at(1) * prior.distribution.at(i).at(j).at(2);
						pro_ijkl.push_back(1 - p1);
						pro_ijkl.push_back(p1);
					}
					pro_ijk.push_back(pro_ijkl);
				}
				pro_ij.push_back(pro_ijk);
			}
			pro_i.push_back(pro_ij);
		}
		probabilities.push_back(pro_i);
	}



	while(t < 10*T)
	{

		bool stop = false;
		vector<Action> actions_2 = Sample_Actions(prior, actions);
		vector<vector<double> > pseudo_distributions;

		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		while((!stop) && t < 10*T)
		{
			int arm = Policy_Whittle_Pseudo(ob, actions_2);
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update3(ob,arm,actions, actions_2);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;
			if(t % T == 0)
				rewards.push_back(reward);

			non_trans_observation.at(arm).at(start_state).at(running_time).at(end_state) += 1;
			int u = non_trans_observation.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation.at(arm).at(start_state).at(running_time).at(1);
			int v = non_trans_observation2.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation2.at(arm).at(start_state).at(running_time).at(1);
			if(u > v * 2)
				stop = true;
		}

		for(int i = 0; i < n; i++)
		{
			double sum_pro = 0;
			vector<double> sums;
			double min = 100000000;
			double max = -100000000;
			for(int j = 0; j < size * size; j++)
			{
				if(prior.distribution.at(i).at(j).at(0) == 0)
				{
					//cout << j << endl;
					sums.push_back(0);
				}
				else
				{
					double sum = 0;
					for(int k = 0; k < m; k++)
					{
						for(int l = 0; l < TMAX; l++)
						{
							for(int l2 = 0; l2 < m; l2++)
							{
								int numbers = non_trans_observation.at(i).at(k).at(l).at(l2) - non_trans_observation2.at(i).at(k).at(l).at(l2);
								sum += numbers * log(probabilities.at(i).at(j).at(k).at(l).at(l2));
							}
						}
					}
					if(sum < min)
						min = sum;
					if(sum > max)
						max = sum;
					sums.push_back(sum);
				}
			}
			//cout << "maxmin" << max << endl << min << endl;
			for(int j = 0; j < size * size; j++)
			{
				double uvw = max - sums.at(j);
				//cout << sums.at(j) << endl;
				//if(uvw > check_max)
				//{
				//	check_max = uvw;
				//	cout << check_max << endl;
				//}
				if(uvw >= MAX_GAP)
					prior.distribution.at(i).at(j).at(0) = 0;
				else if(prior.distribution.at(i).at(j).at(0) != 0)
					prior.distribution.at(i).at(j).at(0) *= exp(MAX_GAP - uvw);
				//cout << prior.distribution.at(i).at(j).at(0) << endl;
				sum_pro = sum_pro + prior.distribution.at(i).at(j).at(0);
			}
			//cout << endl;
			for(int j = 0; j < size * size; j++)
			{
				prior.distribution.at(i).at(j).at(0) /= sum_pro;
				//cout << prior.distribution.at(i).at(j).at(0) << endl;
			}
			//cout << endl;
		}
		//cout << t << endl;

	}
	return rewards;
}

double Game_TS_Pseudo(vector<Action> actions, int T)
{
	double check_max = -100;
	double MAX_GAP = 100;
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;




	while(t < T)
	{

		bool stop = false;
		vector<Action> actions_2 = Init_Actions("GameVirtual1.txt");
		vector<vector<double> > pseudo_distributions;

		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		while((!stop) && t < T)
		{
			int arm = Policy_Whittle_Pseudo(ob, actions_2);
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update3(ob,arm,actions, actions_2);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;

		}
	}
	return reward;
}

double Game_TS_Pseudo2(vector<Action> actions, int T)
{
	double check_max = -100;
	double MAX_GAP = 100;
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;




	while(t < T)
	{

		bool stop = false;
		vector<Action> actions_2 = Init_Actions("GameVirtual2.txt");
		vector<vector<double> > pseudo_distributions;

		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		while((!stop) && t < T)
		{
			int arm = Policy_Whittle_Pseudo(ob, actions_2);
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update3(ob,arm,actions, actions_2);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;
		}
	}
	return reward;
}

double Game_TS_Pseudo3(vector<Action> actions, int T)
{
	double check_max = -100;
	double MAX_GAP = 100;
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;




	while(t < T)
	{

		bool stop = false;
		vector<Action> actions_2 = Init_Actions("GameVirtual3.txt");
		vector<vector<double> > pseudo_distributions;

		for(int i = 0; i < ob.n; i++)
		{
			vector<double> start(actions.at(0).m, 0.0);
			start.at(ob.s.at(i)) = 1.0;
			pseudo_distributions.push_back(Next_distribution(start, Matrix_Power(actions_2.at(i).t, ob.t.at(i), actions_2.at(i))));
		}
		ob.real_distribution_pseudo = pseudo_distributions;
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		while((!stop) && t < T)
		{
			int arm = Policy_Whittle_Pseudo(ob, actions_2);
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update3(ob,arm,actions, actions_2);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;
		}
	}
	return reward;
}

vector<Action> Sample_Actions(Prior_Distribution prior, vector<Action> actions)
{
	vector<Action> sample_actions;
	for(int i = 0; i < prior.n; i++)
	{
		Action action;
		action.m = prior.m;
		action.r = actions.at(i).r;
		action.s0 = 0;
		vector<vector<double> > dis, dis_power;
		double rand = Uniform_Random();
		double sum = 0;
		for(int j = 0; j < prior.size * prior.size; j++)
		{
			sum += prior.distribution.at(i).at(j).at(0);
			if(sum >= rand)
			{
				vector<double> dis1, dis2;
				dis1.push_back(1 - prior.distribution.at(i).at(j).at(1));
				dis1.push_back(prior.distribution.at(i).at(j).at(1));
				dis2.push_back(1 - prior.distribution.at(i).at(j).at(2));
				dis2.push_back(prior.distribution.at(i).at(j).at(2));
				dis.push_back(dis1);
				dis.push_back(dis2);
				break;
			}
		}
		action.t = dis;
		dis_power = dis;
		action.power_record.push_back(dis_power);
		sample_actions.push_back(action);
	}
	return sample_actions;
}

Prior_Distribution Init_Prior(vector<Action> actions, int size)
{
	Prior_Distribution prior;
	prior.n = actions.size();
	prior.m = actions.at(0).m;
	prior.size = size;
	double basic = 1.0/(size+1);
	vector<vector<vector<double> > > distribution;
	for(int i = 0; i < prior.n; i++)
	{
		vector<vector<double> > distribution_i;
		for(int j = 0; j < prior.size; j++)
		{
			for(int k = 0; k < prior.size; k++)
			{
				vector<double> distribution_ij;
				distribution_ij.push_back(1.0/size/size);
				distribution_ij.push_back(basic * (j+1));
				distribution_ij.push_back(basic * (k+1));
				distribution_i.push_back(distribution_ij);
			}
		}
		distribution.push_back(distribution_i);
	}
	prior.distribution = distribution;
	return prior;
}



vector<double> Game_UCRL_vector(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	vector<double> rewards(1,0.0);
	int arm = Policy_default(ob);
	ob = Update2(ob,arm,actions);
	reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
	t = t + 1;

	double delta = 0.001;

	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}

	while(t < 10*T)
	{
		bool stop = false;
		vector<vector<vector<State> > > All_States = Available_States(actions,TMAX-1);
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		double gap = 10000;
		int u = All_States.size();
		int v = All_States.at(0).size();
		int w = All_States.at(0).at(0).size();
		int t_k = 1;
		double max,min;
		/*for(int l = 0; l < u; l++)
		{
			for(int i = 0; i < v; i++)
			{
				for(int t = 0; t < w; t++)
				{
					int state_1 = All_States.at(l).at(i).at(t).states.at(0);
					int state_2 = All_States.at(l).at(i).at(t).states.at(1);
					int time_1 = All_States.at(l).at(i).at(t).times.at(0)-1;
					int time_2 = All_States.at(l).at(i).at(t).times.at(1)-1;
					for(int k = 0; k < 2; k++)
					{
						int numbers1, numbers2;
						if(k == 0)
						{
							numbers1 = non_trans_observation.at(0).at(state_1).at(time_1).at(0);
							numbers2 = non_trans_observation.at(0).at(state_1).at(time_1).at(1);
						}
						else
						{
							numbers1 = non_trans_observation.at(1).at(state_2).at(time_2).at(0);
							numbers2 = non_trans_observation.at(1).at(state_2).at(time_2).at(1);
						}
						if(numbers1 + numbers2 == 0)
						{
							All_States.at(l).at(i).at(t).rewards.at(k) = actions.at(k).r.at(1);
							//All_States.at(l).at(i).at(t).transitions.at(k).at(0) = 0.5;
							//All_States.at(l).at(i).at(t).transitions.at(k).at(1) = 0.5;
						}
						else
						{
							//All_States.at(l).at(i).at(t).rewards.at(k) = (numbers1+0.0)/(numbers1 + numbers2) * actions.at(k).r.at(1) + sqrt(7*log(2*T/delta)/2/(numbers1 + numbers2));
							All_States.at(l).at(i).at(t).rewards.at(k) = (numbers1+0.0)/(numbers1 + numbers2) * actions.at(k).r.at(1) + sqrt(0.01*7*log(2*T/delta)/2/(numbers1 + numbers2));
							if(All_States.at(l).at(i).at(t).rewards.at(k) > 1)
								All_States.at(l).at(i).at(t).rewards.at(k) = 1;
							//All_States.at(l).at(i).at(t).transitions.at(k).at(0) = (numbers1+0.0)/(numbers1 + numbers2);
							//All_States.at(l).at(i).at(t).transitions.at(k).at(1) = (numbers2+0.0)/(numbers1 + numbers2);
						}
					}
				}
			}
		}*/
		

		int iter = 0;
		while(gap > 0.001 && iter <= 100)
		{
			vector<vector<vector<State> > > All_States2(All_States);
			max = -10000;
			min = 10000;
			for(int l = 0; l < u; l++)
			{
				for(int i = 0; i < v; i++)
				{
					for(int tt = 0; tt < w; tt++)
					{
						double V = -10000;
						int state_1 = All_States.at(l).at(i).at(tt).states.at(0);
						int state_2 = All_States.at(l).at(i).at(tt).states.at(1);
						int time_1 = All_States.at(l).at(i).at(tt).times.at(0)-1;
						int time_2 = All_States.at(l).at(i).at(tt).times.at(1)-1;
						for(int k = 0; k < 2; k++)
						{
							double sum = 0;
							double V_0 = All_States.at(l).at(i).at(tt).Next_stat.at(k).at(0)->V_value;
							double V_1 = All_States.at(l).at(i).at(tt).Next_stat.at(k).at(1)->V_value;
							V_1 += actions.at(k).r.at(1);
							V_0 += actions.at(k).r.at(0);
							int numbers1, numbers2;
							if(k == 0)
							{
								numbers1 = non_trans_observation.at(0).at(state_1).at(time_1).at(0);
								numbers2 = non_trans_observation.at(0).at(state_1).at(time_1).at(1);
							}
							else
							{
								numbers1 = non_trans_observation.at(1).at(state_2).at(time_2).at(0);
								numbers2 = non_trans_observation.at(1).at(state_2).at(time_2).at(1);
							}
							if(numbers1 + numbers2 == 0)
								sum = V_0 > V_1 ? V_0 : V_1;
							else
							{
								if(V_0 > V_1)
								{
									double p_0 = (numbers1+0.0)/(numbers1+numbers2);
									p_0 = p_0 + sqrt(56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									//p_0 = p_0 + sqrt(0.01*56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									if(p_0 > 1)
										p_0 = 1;
									sum = p_0 * V_0 + (1-p_0) * V_1;
									//sum = (sum + (p_0 * V_0 + (1-p_0) * V_1) * iter)/(iter + 1);
								}
								else
								{
									double p_1 = (numbers2+0.0)/(numbers1+numbers2);
									p_1 = p_1 + sqrt(56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									//p_1 = p_1 + sqrt(0.01*56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									if(p_1 > 1)
										p_1 = 1;
									sum = p_1 * V_1 + (1-p_1) * V_0;
									//sum = (sum + (p_1 * V_1 + (1-p_1) * V_0) * iter)/(iter + 1);
								}
							}
							if(sum > V)
							{
								All_States.at(l).at(i).at(tt).action = k;
								V = sum;
							}
							//cout << sum << endl;
						}
						All_States2.at(l).at(i).at(tt).V_value = V;
					}
				}
			}
			for(int l = 0; l < u; l++)
			{
				for(int i = 0; i < v; i++)
				{
					for(int tt = 0; tt < w; tt++)
					{
						double Vprime = All_States.at(l).at(i).at(tt).V_value;
						double V = All_States2.at(l).at(i).at(tt).V_value;
						All_States.at(l).at(i).at(tt).V_value = V;
						double gap2 = Vprime - V;
						if(gap2 > max)
							max = gap2;
						if(gap2 < min)
							min = gap2;
					}
				}
			}
			gap = max - min;
			iter = iter + 1;
			//cout << gap << endl;
			
		}
		//cout << (max + min)/2 << endl;
		while((!stop) && t < 10*T)
		{
			int l,sprime,timeprime;
			if(ob.t.at(0) == 1)
			{
				l = ob.s.at(0);
				sprime = ob.s.at(1);
				timeprime = ob.t.at(1)-2;
			}
			else
			{
				l = 2 + ob.s.at(1);
				sprime = ob.s.at(0);
				timeprime = ob.t.at(0)-2;
			}
			if(timeprime >= TMAX - 1)
				timeprime = TMAX - 2;
			State actionsss = All_States.at(l).at(sprime).at(timeprime);
			int arm = actionsss.action;
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update2(ob,arm,actions);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;
			if(t % T == 0)
				rewards.push_back(reward);

			non_trans_observation.at(arm).at(start_state).at(running_time).at(end_state) += 1;
			int u = non_trans_observation.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation.at(arm).at(start_state).at(running_time).at(1);
			int v = non_trans_observation2.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation2.at(arm).at(start_state).at(running_time).at(1);
			if(u > v * 2)
				stop = true;
		}
		t_k = t;
		//cout << t << endl;
	}
	return rewards;
}


double Game_UCRL(vector<Action> actions, int T)
{
	Observation ob = Init(actions);
	int t = 0;
	double reward = 0;
	int arm = Policy_default(ob);
	ob = Update2(ob,arm,actions);
	reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
	t = t + 1;

	double delta = 0.001;

	int TMAX = 10;
	vector<vector<vector<vector<int> > > > non_trans_observation;
	int n = actions.size();
	int m = actions.at(0).m;
	for(int i = 0; i < n; i++)
	{
		vector<vector<vector<int> > > non_i;
		for(int j = 0; j < m; j++)
		{
			vector<vector<int> > non_ij;
			for(int k = 0; k < TMAX; k++)
			{
				vector<int> non_ijk(m,0);
				non_ij.push_back(non_ijk);
			}
			non_i.push_back(non_ij);
		}
		non_trans_observation.push_back(non_i);
	}

	while(t < T)
	{
		bool stop = false;
		vector<vector<vector<State> > > All_States = Available_States(actions,TMAX-1);
		vector<vector<vector<vector<int> > > > non_trans_observation2 = non_trans_observation;
		double gap = 10000;
		int u = All_States.size();
		int v = All_States.at(0).size();
		int w = All_States.at(0).at(0).size();
		int t_k = 1;
		double max,min;
		/*for(int l = 0; l < u; l++)
		{
			for(int i = 0; i < v; i++)
			{
				for(int t = 0; t < w; t++)
				{
					int state_1 = All_States.at(l).at(i).at(t).states.at(0);
					int state_2 = All_States.at(l).at(i).at(t).states.at(1);
					int time_1 = All_States.at(l).at(i).at(t).times.at(0)-1;
					int time_2 = All_States.at(l).at(i).at(t).times.at(1)-1;
					for(int k = 0; k < 2; k++)
					{
						int numbers1, numbers2;
						if(k == 0)
						{
							numbers1 = non_trans_observation.at(0).at(state_1).at(time_1).at(0);
							numbers2 = non_trans_observation.at(0).at(state_1).at(time_1).at(1);
						}
						else
						{
							numbers1 = non_trans_observation.at(1).at(state_2).at(time_2).at(0);
							numbers2 = non_trans_observation.at(1).at(state_2).at(time_2).at(1);
						}
						if(numbers1 + numbers2 == 0)
						{
							All_States.at(l).at(i).at(t).rewards.at(k) = actions.at(k).r.at(1);
							//All_States.at(l).at(i).at(t).transitions.at(k).at(0) = 0.5;
							//All_States.at(l).at(i).at(t).transitions.at(k).at(1) = 0.5;
						}
						else
						{
							//All_States.at(l).at(i).at(t).rewards.at(k) = (numbers1+0.0)/(numbers1 + numbers2) * actions.at(k).r.at(1) + sqrt(7*log(2*T/delta)/2/(numbers1 + numbers2));
							All_States.at(l).at(i).at(t).rewards.at(k) = (numbers1+0.0)/(numbers1 + numbers2) * actions.at(k).r.at(1) + sqrt(0.01*7*log(2*T/delta)/2/(numbers1 + numbers2));
							if(All_States.at(l).at(i).at(t).rewards.at(k) > 1)
								All_States.at(l).at(i).at(t).rewards.at(k) = 1;
							//All_States.at(l).at(i).at(t).transitions.at(k).at(0) = (numbers1+0.0)/(numbers1 + numbers2);
							//All_States.at(l).at(i).at(t).transitions.at(k).at(1) = (numbers2+0.0)/(numbers1 + numbers2);
						}
					}
				}
			}
		}*/
		

		int iter = 0;
		while(gap > 0.001 && iter <= 100)
		{
			vector<vector<vector<State> > > All_States2(All_States);
			max = -10000;
			min = 10000;
			for(int l = 0; l < u; l++)
			{
				for(int i = 0; i < v; i++)
				{
					for(int tt = 0; tt < w; tt++)
					{
						double V = -10000;
						int state_1 = All_States.at(l).at(i).at(tt).states.at(0);
						int state_2 = All_States.at(l).at(i).at(tt).states.at(1);
						int time_1 = All_States.at(l).at(i).at(tt).times.at(0)-1;
						int time_2 = All_States.at(l).at(i).at(tt).times.at(1)-1;
						for(int k = 0; k < 2; k++)
						{
							double sum = 0;
							double V_0 = All_States.at(l).at(i).at(tt).Next_stat.at(k).at(0)->V_value;
							double V_1 = All_States.at(l).at(i).at(tt).Next_stat.at(k).at(1)->V_value;
							V_1 += actions.at(k).r.at(1);
							V_0 += actions.at(k).r.at(0);
							int numbers1, numbers2;
							if(k == 0)
							{
								numbers1 = non_trans_observation.at(0).at(state_1).at(time_1).at(0);
								numbers2 = non_trans_observation.at(0).at(state_1).at(time_1).at(1);
							}
							else
							{
								numbers1 = non_trans_observation.at(1).at(state_2).at(time_2).at(0);
								numbers2 = non_trans_observation.at(1).at(state_2).at(time_2).at(1);
							}
							if(numbers1 + numbers2 == 0)
								sum = V_0 > V_1 ? V_0 : V_1;
							else
							{
								if(V_0 > V_1)
								{
									double p_0 = (numbers1+0.0)/(numbers1+numbers2);
									p_0 = p_0 + sqrt(56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									//p_0 = p_0 + sqrt(0.01*56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									if(p_0 > 1)
										p_0 = 1;
									sum = p_0 * V_0 + (1-p_0) * V_1;
									//sum = (sum + (p_0 * V_0 + (1-p_0) * V_1) * iter)/(iter + 1);
								}
								else
								{
									double p_1 = (numbers2+0.0)/(numbers1+numbers2);
									p_1 = p_1 + sqrt(56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									//p_1 = p_1 + sqrt(0.01*56*2*log(4*t_k/delta)/2/(numbers1+numbers2));
									if(p_1 > 1)
										p_1 = 1;
									sum = p_1 * V_1 + (1-p_1) * V_0;
									//sum = (sum + (p_1 * V_1 + (1-p_1) * V_0) * iter)/(iter + 1);
								}
							}
							if(sum > V)
							{
								All_States.at(l).at(i).at(tt).action = k;
								V = sum;
							}
							//cout << sum << endl;
						}
						All_States2.at(l).at(i).at(tt).V_value = V;
					}
				}
			}
			for(int l = 0; l < u; l++)
			{
				for(int i = 0; i < v; i++)
				{
					for(int tt = 0; tt < w; tt++)
					{
						double Vprime = All_States.at(l).at(i).at(tt).V_value;
						double V = All_States2.at(l).at(i).at(tt).V_value;
						All_States.at(l).at(i).at(tt).V_value = V;
						double gap2 = Vprime - V;
						if(gap2 > max)
							max = gap2;
						if(gap2 < min)
							min = gap2;
					}
				}
			}
			gap = max - min;
			iter = iter + 1;
			//cout << gap << endl;
			
		}
		//cout << (max + min)/2 << endl;
		while((!stop) && t < T)
		{
			int l,sprime,timeprime;
			if(ob.t.at(0) == 1)
			{
				l = ob.s.at(0);
				sprime = ob.s.at(1);
				timeprime = ob.t.at(1)-2;
			}
			else
			{
				l = 2 + ob.s.at(1);
				sprime = ob.s.at(0);
				timeprime = ob.t.at(0)-2;
			}
			if(timeprime >= TMAX - 1)
				timeprime = TMAX - 2;
			State actionsss = All_States.at(l).at(sprime).at(timeprime);
			int arm = actionsss.action;
			//cout << arm << endl;
			int start_state = ob.s.at(arm);
			int running_time = ob.t.at(arm) - 1;
			if(running_time >= TMAX)
				running_time = TMAX - 1;
			ob = Update2(ob,arm,actions);
			int end_state = ob.s.at(arm);
			reward += Bernoulli(actions.at(arm).r.at(ob.s.at(arm)));
			t = t + 1;

			non_trans_observation.at(arm).at(start_state).at(running_time).at(end_state) += 1;
			int u = non_trans_observation.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation.at(arm).at(start_state).at(running_time).at(1);
			int v = non_trans_observation2.at(arm).at(start_state).at(running_time).at(0) + non_trans_observation2.at(arm).at(start_state).at(running_time).at(1);
			if(u > v * 2)
				stop = true;
		}
		t_k = t;
		//cout << t << endl;
	}
	return reward;
}


vector<vector<double> > Regret_ReUCB(vector<Action> actions, int T, int number, double best)
{
	vector<vector<double> > result;
	vector<double> regrets(1,0.0);
	vector<double> variance(1,0.0);
	for(int i = 0; i < 10; i++)
	{
		double sum = 0;
		double v = 0;
		for(int j = 0; j < number; j++)
		{
			double regret = (best * T * (i+1) - Game_ReUCB(actions, T*(i+1)));
			sum += regret/number;
			v += regret * regret / number;
		}
		regrets.push_back(sum);
		variance.push_back(sqrt(v - sum * sum));
	}
	result.push_back(regrets);
	result.push_back(variance);
	return result;


}

vector<vector<double> > Regret_TS(vector<Action> actions, int T, int size, int number, double best)
{
	vector<vector<double> > result;
	vector<double> regrets(1,0.0);
	vector<double> variance(1,0.0);
	for(int i = 0; i < 10; i++)
	{
		double sum = 0;
		double v = 0;
		for(int j = 0; j < number; j++)
		{
			double regret = (best * T * (i+1) - Game_TS(actions, T*(i+1), size));
			sum += regret/number;
			v += regret * regret / number;
		}
		regrets.push_back(sum);
		variance.push_back(sqrt(v - sum * sum));
	}
	result.push_back(regrets);
	result.push_back(variance);
	return result;


}

vector<vector<double> > Regret_TS_vector(vector<Action> actions, int T, int size, int number, double best)
{
	vector<vector<double> > result;
	vector<double> regrets(11,0.0);
	vector<double> variance(11,0.0);
	for(int j = 0; j < number; j++)
	{
		vector<double> rewards = Game_TS_vector(actions, T, size);
		for(int i = 0; i < 11; i++)
		{
			regrets.at(i) += (best * T * i - rewards.at(i))/number;
			variance.at(i) += (best * T * i - rewards.at(i)) * (best * T * i - rewards.at(i))/number;
		}	
	}
	for(int i = 0; i < 11; i++)
		variance.at(i) = sqrt(variance.at(i) - regrets.at(i)*regrets.at(i));
	result.push_back(regrets);
	result.push_back(variance);
	return result;


}



vector<vector<double> > Regret_UCRL(vector<Action> actions, int T, int number, double best)
{
	vector<vector<double> > result;
	vector<double> regrets(1,0.0);
	vector<double> variance(1,0.0);
	for(int i = 0; i < 10; i++)
	{
		double sum = 0;
		double v = 0;
		for(int j = 0; j < number; j++)
		{
			double regret = (best * T * (i+1) - Game_UCRL(actions, T*(i+1)));
			sum += regret/number;
			v += regret * regret / number;
		}
		regrets.push_back(sum);
		variance.push_back(sqrt(v - sum * sum));
	}
	result.push_back(regrets);
	result.push_back(variance);
	return result;


}

vector<vector<double> > Regret_UCRL_vector(vector<Action> actions, int T, int number, double best)
{
	vector<vector<double> > result;
	vector<double> regrets(11,0.0);
	vector<double> variance(11,0.0);
	for(int j = 0; j < number; j++)
	{
		vector<double> rewards = Game_UCRL_vector(actions, T);
		for(int i = 0; i < 11; i++)
		{
			regrets.at(i) += (best * T * i - rewards.at(i))/number;
			variance.at(i) += (best * T * i - rewards.at(i)) * (best * T * i - rewards.at(i))/number;
		}	
	}
	for(int i = 0; i < 11; i++)
		variance.at(i) = sqrt(variance.at(i) - regrets.at(i)*regrets.at(i));
	result.push_back(regrets);
	result.push_back(variance);
	return result;


}

vector<vector<double> > Regret_UCRL_vector_M(vector<Action> actions, int T, int number, double best)
{
	vector<double> regrets(11,0.0);
        vector<double> variance(11, 0.0);
	for(int j = 0; j < number; j++)
	{
		vector<double> rewards = Game_UCRL_vector_M(actions, T);
		for(int i = 0; i < 11; i++)
		{
			regrets.at(i) += (best * T * i - rewards.at(i))/number;
			variance.at(i) += (best * T * i - rewards.at(i))*(best * T * i - rewards.at(i))/number;
		}
	}
	for(int i = 0; i < 11; i++)
		variance.at(i) = sqrt(variance.at(i) - regrets.at(i) * regrets.at(i));
        vector<vector<double> > r;
        r.push_back(regrets);
        r.push_back(variance);
	return r;

}
















