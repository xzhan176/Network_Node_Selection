# Opinion Polarization Games over Social Network
Two players a maximizer and a minimizer change agent's opinion in the network. Maximizer trys to select a agent and change its opinion so that the polarization at equilibrium will be the maximum. Minimizer trys to take an action(select an agent, and change its opinion to a value) to minimize the equilibrium polarization of the network. 
Abstract ...

This paper includes sythetic network and real network. 
  Systhetic network is generated in the python code. 
  Real networks include Reddit and Twitter network that imported from read datasets. 
All datasets are in the Data folder. 

# Python Code
This code has been written in Python. All neccessary packages have been included in the code. This directory contains the following documented Python files. 

1. Extreme Network 1: Given a graph represented by the adjacency matrix, a vector of innate opinions(1, 0.5, 0), and the code for fictitious play that two players(Maximizer and Minimizer) play on the network until a Nash Equilibrium result is found.
2. Extreme Network 2: Given a symmetric triangle network, except the network structure, other are same as Extrem Network 1.
3. Testing Reddit: Imported Reddit network from data folder, then two players start fictitious play on the Reddit network until a Nash Equlibrium is found.
4. Testing Twitter: Imported Twitter network from data folder, same procedure as Reddit Network. 
5. pure_startegy_selection: this code contains functions to execute first-round greedy action for player 1 and player 2. It is called in max_fir_play() and min_fir_play() function if they are used.

## Running an Experiemnt 
### Initial Conditions
At the first round, two players choose an action without considering another player's action. 

Above python code start the game with players' random actions in the first round by calling Random_play() function. Every time one run the expriment in the above 1-4 
python code, the random_play() will generate random actions and overwrite last random action. Two players randomly choose an agent and change it's innate opnion to 0 
or 1 without considering the effect to polarization. 

The game can also start with players' intentional selection(i.e., in the first round, player 1 choose the action that maximize polarization, player 2 choose the action 
that minimize polarization). To achieve this, comment out the line (v1, max_opinion) = random_play() at line and line, then uncomment (v1, max_opinion) = max_first_play() 
at line xxx and (v2, min_opinion) = min_first_play() at line xxx.

### Innate Opinons and Network Structure
One can change the innate opinion/structure of the network at the Section 2 in the code "Innate Opinions and Adjacency Matrix".

### Exit Out Situation
If two players selected same agent at the first round, the game will exit out automatically. Since two players cannot choose the same agent in same round based on Game
theory principal, the conditional Exit Out command is to avoid this happens. If one come across Exit Out situation, just run the game again, then the new random 
actions will overwrite the previous actions. 
