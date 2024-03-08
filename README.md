
# Opinion Polarization Games over Social Networks
Authors: Xilin Zhang, Emrah Akyol, Zeynep Ertem, xxx
## Abstract 
This paper provides a quantitative analysis of a game over a social network of agents, some of which are controlled by two players whose objectives are to maximize and minimize respectively polarization over this network. Opinions of agents evolve according to the Friedkin-Johnsen model, and players can change only the innate opinion of an agent of their choosing. Polarization is measured via the sample variance of the agents' steady-state opinions. The practically motivated constraint on the set of players' choice of agents to be disjoint transforms this simple zero-sum game into a compelling and largely unexplored research problem. We first analyze the functional properties of this game and characterize the optimal best response for each player given the agent. Next, we analyze the properties of the Nash equilibrium. Finally, we simulate a variation of the fictitious play algorithm to obtain the equilibrium in synthetic and real data networks, where the constraint mentioned above imposes a minor modification on the classical fictitious play algorithm. All of our codes and datasets are available publicly for research purposes. 


## Python Code
This code has been written in Python. All necessary packages have been included in the code. This directory contains the following documented Python files. 

1. Final_nzs_Fictitious_Play: Imported network data, then two players start fictitious play where they cannot choose the same agents that are in another player's territory.
2. Final_zs_Fictitious_Play: Import network data, then two players start fictitious play where they can choose the same agent in the network, but then both players' action effects will be canceled off.
4. Final_MaxMin Test: Imported real network from the data folder, then start the Stackelberg game. Maximizer knows all minimizer's actions corresponding to each minimizer's action.  
5. Final_MaxMin Test: Imported real network from the data folder, then start the Stackelberg game. Minimizer knows all maximizer's actions corresponding to each maximizer's action.
6. pure_startegy_selection: this code contains functions to execute first-round greedy action for player 1 and player 2. It is called in max_fir_play() and min_fir_play() functions if they are used. No need to open it for running an experiment.

## Running an Experiment 
The game requires several hyper-parameters that are listed below. An example of all the hyper-parameters used in the game is included in the Python Code folder.

[Game] <br>
Players = 2; This algorithm designed the game only for 2 players <br>
PayoffMatrix = True; True: save payoff matrix; False: save the network data with the label (selected nodes), not payoff matrix <br>
GameType = 1; 1: Non-zero-sum game; 2: Zero-sum game; 3: Maximin Stackelberg game; 4: Minimax Stackelberg game<br>

[Algorithm] <br>
ConvergenceThreshold = 0.01 <br>
MaxIterations(K) = 1000 <br>

[InitialConditions] <br>
Each game starts with a random action.<br>
PlayerStrategies = action1(for Maximizer), actionA(for Minimizer); It can be random or a fixed starting strategy <br>
PlayerPayoffs = call payoff function <br>

[Output] <br>
SaveIntermediateResults = true <br>
SaveConvergenceInfo = true <br>

If any problem occurs while running the code, please contact us at xzhan176@binghamton.edu.


