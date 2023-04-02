
# Opinion Polarization Games over Social Network
Authors: Emrah Akyol, Xilin Zhang, Zeynep Ertem
## Abstract 
This paper provides a quantitative analysis of a game over a social network of agents, some of which are controlled by two players whose objectives are to maximize and minimize respectively polarization over this network. Opinions of agents evolve according to the Friedkin-Johnsen model, and players can change only the innate opinion of an agent of their choosing. Polarization is measured via the sample variance of the agents' steady-state opinions. The practically motivated constraint on the set of players' choice of agents to be disjoint transforms this simple zero-sum game into a compelling and largely unexplored research problem. We first analyze the functional properties of this game and characterize the optimal best response for each player given the agent. Next, we analyze the properties of the Nash equilibrium. Finally, we simulate a variation of the fictitious play algorithm to obtain the equilibrium in synthetic and real data networks, where the constraint mentioned above imposes a minor modification on the classical fictitious play algorithm. All of our codes and datasets are available publicly for research purposes. 


## Python Code
This code has been written in Python. All neccessary packages have been included in the code. This directory contains the following documented Python files. 

1. Testing Extreme Network 1: Given a graph represented by the adjacency matrix, a vector of innate opinions(1, 0.5, 0), and the code for fictitious play that two players(Maximizer and Minimizer) play on the network until a Nash Equilibrium result is found.
2. Testing Extreme Network 2: Given a symmetric triangle network, except the network structure, other features are same as Extrem Network 1.
3. Testing Reddit: Imported Reddit network from data folder, then two players start fictitious play on the Reddit network until a Nash Equlibrium is found.
4. Testing Twitter: Imported Twitter network from data folder, same procedure as Reddit Network. 
5. Testing Karate: Imported Karate network from data folder, same procedure as Reddit Network. 
6. pure_startegy_selection: this code contains functions to execute first-round greedy action for player 1 and player 2. It is called in max_fir_play() and min_fir_play() function if they are used.

## Running an Experiemnt 
### 1. Initial Conditions
At the first round, two players choose an action without considering another player's action. 

Above python code start the game with players' random actions in the first round by calling Random_play() function. Every time one run the expriment in the above 1-4 
python code, the random_play() will generate random actions and overwrite last random action. Two players randomly choose an agent and change it's innate opnion to 0 
or 1 without considering the effect to polarization. 

The game can also start with players' intentional selection(i.e., in the first round, player 1 choose the action that maximize polarization, player 2 choose the action 
that minimize polarization). To achieve this, comment out the line (v1, max_opinion) = random_play() at line and line, then uncomment (v1, max_opinion) = max_first_play() 
at line xxx and (v2, min_opinion) = min_first_play() at line xxx.

### 2. Innate Opinons and Network Structure
One can change the innate opinion/structure of the network at the Section 2 in the code "Innate Opinions and Adjacency Matrix".

### 3. Exit Out Situation
If two players selected same agent at the first round, the game will exit out automatically. Since two players cannot choose the same agent in same round based on Game
theory principal, the conditional Exit Out command is to avoid this happens. If one come across Exit Out situation, just run the game again, then the new random 
actions will overwrite the previous actions. 
