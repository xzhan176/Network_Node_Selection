
# Opinion Polarization Games over Social Networks
Authors: Xilin Zhang, Emrah Akyol, Zeynep Ertem
## Abstract 
This paper provides a quantitative analysis of a game over a social network of agents, some of which are controlled by two players whose objectives are to maximize and minimize respectively polarization over this network. Opinions of agents evolve according to the Friedkin-Johnsen model, and players can change only the innate opinion of an agent of their choosing. Polarization is measured via the sample variance of the agents' steady-state opinions. We first analyze the functional properties of this game and characterize the optimal best response for each player given the agent. Next, we analyze the properties of the Nash equilibrium. Finally, we simulate a variation of the fictitious play algorithm to obtain the Nash equilibrium on the Karate Club and Reddit networks. All of our codes and datasets are available publicly for research purposes. 


## Python Code
This code has been written in Python. All necessary packages have been included in the code. This directory contains the following documented Python files. 

1. Runfile-ZeroSum_Fictitious_Play.ipynb: Read network data, then two players(Maximizer and Minimzier) start fictitious play, the algorithm is explained in the paper. 
2. Karate.ipynb: Python code importing Karate club network data. 
3. Reddit.ipynb: Python code importing Reddit network data. 


## Running an Experiment 
The game requires several hyper-parameters that are listed below. An example of all the hyper-parameters used in the game is included in the Python Code folder.

[InitialConditions] <br>
Each game starts with a random action.<br>
PlayerStrategies = action1(for Maximizer), actionA(for Minimizer); It can be random or a fixed starting strategy <br>
PlayerPayoffs = call payoff function <br>

[Output] <br>
SavePayoffMatrix = true <br>
SaveConvergenceInfo = true <br>

If any problem occurs while running the code, please contact us at xzhan176@binghamton.edu.


