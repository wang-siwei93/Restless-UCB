# Restless-UCB

This is the main function of Game.cpp, minor changes on the parameters can reproduce those experimental results.

int main()
{
	srand((unsigned)time(NULL));  
	vector<Action> actions = Init_Actions("Instances/Construct_Instance_1.txt");//Read the instance file
	double best_upper_bound = Best_Average(actions); //An upper bound for the average reward of the optimal policy
  double best_oracle = Game_Whittle(actions, 10000000)/10000000; //An average of the offline oracle
        //
	vector<vector<double> > regrets_ReUCB = Regret_ReUCB(actions, 5000, 1000, best_oracle);//The Restless-UCB policy with T = 5000 to 50000 and 1000 runs
	print_vector(regrets_ReUCB[0]);//The average regrets
	print_vector(regrets_ReUCB[1]);//The variance of regrets
	//
	vector<vector<double> > regrets_TS9 = Regret_TS_vector(actions, 5000, 9, 1000, best_oracle);//The TS policy with T = 5000 to 50000 and 1000 runs, its support on transition matrices are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]^2
	print_vector(regrets_TS9[0]);//The average regrets
	print_vector(regrets_TS9[1]);//The variance of regrets
	//
	vector<vector<double> > regrets_TS4 = Regret_TS_vector(actions, 5000, 4, 1000, best_oracle);//The TS policy with T = 5000 to 50000 and 1000 runs, its support on transition matrices are [0.2, 0.4, 0.6, 0.8]^2
	print_vector(regrets_TS4[0]);//The average regrets
	print_vector(regrets_TS4[1]);//The variance of regrets
	//
	vector<vector<double> > regrets_UCRL = Regret_UCRL_vector(actions, 5000, 1000, best_oracle);//The Colored-UCRL policy with T = 5000 to 50000 and 1000 runs
	print_vector(regrets_UCRL[0]);//The average regrets
	print_vector(regrets_UCRL[1]);//The variance of regrets
	//
	//Only use the following commands for colored-UCRL2 when there are more than 2 arms
	//vector<vector<double> > regrets_UCRL = Regret_UCRL_vector_M(actions, 5000, 100, best_oracle); //The Colored-UCRL policy with T = 5000 to 50000 and 100 runs
	//print_vector(regrets_UCRL[0]);
  //print_vector(regrets_UCRL[1]);
	return 0;
}
