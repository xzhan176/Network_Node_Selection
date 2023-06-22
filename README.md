
# Opinion Polarization Games over Social Networks
Authors: Xilin Zhang, Emrah Akyol, Zeynep Ertem
## Abstract 
This paper provides a quantitative analysis of a game over a social network of agents, some of which are controlled by two players whose objectives are to maximize and minimize respectively polarization over this network. Opinions of agents evolve according to the Friedkin-Johnsen model, and players can change only the innate opinion of an agent of their choosing. Polarization is measured via the sample variance of the agents' steady-state opinions. The practically motivated constraint on the set of players' choice of agents to be disjoint transforms this simple zero-sum game into a compelling and largely unexplored research problem. We first analyze the functional properties of this game and characterize the optimal best response for each player given the agent. Next, we analyze the properties of the Nash equilibrium. Finally, we simulate a variation of the fictitious play algorithm to obtain the equilibrium in synthetic and real data networks, where the constraint mentioned above imposes a minor modification on the classical fictitious play algorithm. All of our codes and datasets are available publicly for research purposes. 


## Python Code
This code has been written in Python. All necessary packages have been included in the code. This directory contains the following documented Python files. 

1. Final_nzs_Fictitious_Play: Imported network data, then two players start fictitious play where they cannot choose the same agents that are in another player's territory.
2. Final_zs_Fictitious_Play: Import network data, then two players start fictitious play where they can choose the same agent in the network, but then both players' action effects will be canceled off.
4. Final_MaxMin Test: Imported real network from the data folder, then start the Stackelberg game. Maximizer knows all minimizer's actions corresponding to each minimizer's action.  
5. Final_MaxMin Test: Imported real network from the data folder, then start the Stackelberg game. Minimizer knows all maximizer's actions corresponding to each maximizer's action.
6. Final_k_node_Fictitious_Play: two players start fictitious play, and each player can choose k nodes.
7. Testing Extreme Network 1: Given a graph represented by the adjacency matrix, a vector of innate opinions(1, 0.5, 0), and the code for fictitious play that two players(Maximizer and Minimizer) play on the network until a Nash Equilibrium result is found.
8. Testing Extreme Network 2: Given a symmetric triangle network, except for the network structure, other features are the same as Extrem Network 1.
9. pure_startegy_selection: this code contains functions to execute first-round greedy action for player 1 and player 2. It is called in max_fir_play() and min_fir_play() function if they are used.

## Running an Experiment 
The game requires several hyper-parameters that are listed below. An example of using a configuration file to pass all the following parameters to the game is included in the Python Code folder.

[Game] <br>
Players = 2; This algorithm designed the game only for 2 players <br>
PayoffMatrix = True; True: save payoff matrix; False: save the network data with label (selected nodes), not payoff matrix <br>
Type = 1; 1: Non-zero-sum game; 2: Zero-sum game; 3: Maximin Stackelberg game; 4: Minimax Stackelberg game<br>
payoff_function = 1; 1: P(z) = α Polarization(z) + (1-α) Disagreement(z); 2: customized payoff function <br>
α = 1; α is the ratio parameter in the payoff function, can be any value between 0 and 1<br>

[Algorithm] <br>
ConvergenceThreshold = 0.01 <br>
MaxIterations = 1000 <br>

[Strategies] <br>
\kappa_1 = θ1  ; Number of nodes player 1 can select at each pure action <br>
\kappa_2 = θ1  ; Number of nodes player 2 can select at each pure action <br>

[InitialConditions] <br>
PlayerStrategies = strategy1, strategyA; It can be random or a fixed starting strategy <br>
PlayerPayoffs = call payoff function <br>

[Output] <br>
SaveIntermediateResults = true <br>
SaveConvergenceInfo = true <br>

If any problem occurs while running the code, please contact us at xzhan176@binghamton.edu.
## k Nodes Selection Algorithm
![image](https://github.com/xzhan176/Network_Node_Selection/assets/73297832/28d0d57f-f8a6-4f5d-9ddf-6498a67b7ce3)

