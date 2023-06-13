
# Opinion Polarization Games over Social Networks
Authors: Xilin Zhang, Emrah Akyol, Zeynep Ertem
## Abstract 
This paper provides a quantitative analysis of a game over a social network of agents, some of which are controlled by two players whose objectives are to maximize and minimize respectively polarization over this network. Opinions of agents evolve according to the Friedkin-Johnsen model, and players can change only the innate opinion of an agent of their choosing. Polarization is measured via the sample variance of the agents' steady-state opinions. The practically motivated constraint on the set of players' choice of agents to be disjoint transforms this simple zero-sum game into a compelling and largely unexplored research problem. We first analyze the functional properties of this game and characterize the optimal best response for each player given the agent. Next, we analyze the properties of the Nash equilibrium. Finally, we simulate a variation of the fictitious play algorithm to obtain the equilibrium in synthetic and real data networks, where the constraint mentioned above imposes a minor modification on the classical fictitious play algorithm. All of our codes and datasets are available publicly for research purposes. 


## Python Code
This code has been written in Python. All neccessary packages have been included in the code. This directory contains the following documented Python files. 

1. Testing Extreme Network 1: Given a graph represented by the adjacency matrix, a vector of innate opinions(1, 0.5, 0), and the code for fictitious play that two players(Maximizer and Minimizer) play on the network until a Nash Equilibrium result is found.
2. Testing Extreme Network 2: Given a symmetric triangle network, except the network structure, other features are same as Extrem Network 1.
3. Final_Fictitious Play: Imported real network from data folder, then two players start fictitious play on the Reddit network until a Nash Equlibrium is found.
4. Final_k_node_Fictitious_Play: two player start fictitious play, each player can choose k nodes.
5. Final_MaxMin Test: Imported real network from data folder, then start the stackelberg game. Maximizer knows all minimizer's action.  
6. Final_MaxMin Test: Imported real network from data folder, then start the stackelberg game. Minimizer knows all maximizer's action.  
7. pure_startegy_selection: this code contains functions to execute first-round greedy action for player 1 and player 2. It is called in max_fir_play() and min_fir_play() function if they are used.

## Running an Experiemnt 
[Game]
Players = 2
PayoffMatrix = True  ; True: save payoff matrix; False: save the network data with label (selected nodes), not payoff matrix <br>
Type = 1  ; 1: Non-zero-sum game; 2: Zero-sum game; 3: Maximin Stackelberg game; 4: Minimax Stackelberg game<br>
payoff_function = 1  ; 1: P(z) = α Polarization(z) + (1-α) Disagreement(z); 2: customized payoff function <br>
α = 1  ; α is the ratio parameter in the payoff function, can be any value between 0 and 1<br>

[Algorithm] <br>
ConvergenceThreshold = 0.01 <br>
MaxIterations = 1000 <br>
LearningRate = 0.1 <br>

[Strategies] <br>
Player1 = θ1  ; Number of nodes player 1 can select at each pure action <br>
Player2 = θ1  ; Number of nodes player 2 can select at each pure action <br>

[InitialConditions] <br>
PlayerStrategies = strategy1, strategyA  ; It can be random or a fixed starting strategy <br>
PlayerPayoffs = call payoff function <br>

[Output] <br>
SaveIntermediateResults = true <br>
SaveConvergenceInfo = true <br>

[Execution] <br>
ParallelComputation = false <br>
RandomSeed = 12345 <br>
RuntimeLimit = 60 <br>

