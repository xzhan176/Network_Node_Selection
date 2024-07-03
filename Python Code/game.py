import numpy as np
import collections
import sys
from utils import *
from gameutils import *


class GameResult:
    def __init__(self, first_max, first_min, min_touched, max_touched,  payoff_matrix, min_touched_last_100, min_touched_all, fla_min_fre, fla_max_fre, min_history, max_history, min_history_last_100, max_history_last_100):
        self.first_max = first_max
        self.first_min = first_min
        self.min_touched = min_touched
        self.max_touched = max_touched
        self.payoff_matrix = payoff_matrix
        self.min_touched_last_100 = min_touched_last_100
        self.min_touched_all = min_touched_all
        self.fla_min_fre = fla_min_fre
        self.fla_max_fre = fla_max_fre
        self.min_history = min_history
        self.max_history = max_history
        self.min_history_last_100 = min_history_last_100
        self.max_history_last_100 = max_history_last_100


class Game:
    def run(self, game_rounds, memory, s, n, A, L):
        op = copy.copy(s)
        payoff_matrix = np.empty((0, 2*n), float)
        max_touched = []
        min_touched = []
        min_touched_all = []
        min_touched_last_100 = []
        max_history = np.zeros([n, 2])  # n*2 matrix, agent i & opinion options
        min_history = []
        max_history_last_100 = np.zeros([n, 2])
        min_history_last_100 = []

        # Game start from maximizer random play
        # Maximizer start with random action
        v1, max_opinion, max_pol = random_play(op, n, A, L)

        # Save the first action to report later
        first_max = (v1, max_opinion, max_pol)

        # save Maximizer's action history
        max_touched.append(v1)

        # store maximizer play history, using agent(row) and changed opinion(column) as indicator to locate history
        max_history[v1, int(max_opinion)] += 1

        # its frequency, only played 1 time so far, divided by 1
        max_frequency = max_history / 1

        column = column_index(v1, max_opinion)

        # flatten the n*2 matrix to a 2n*1 matrix
        fla_max_fre = max_frequency.flatten()
        print(f'fre_max at spot: {fla_max_fre[column]}')

        # the frequency of maximizer's most recent action (v1,max_opinion)

        # if game start from minimizer random play - make sure two random play are not same agent!!!
        while True:
            retry = 0
            v2, min_opinion, min_pol = random_play(op, n, A, L)
            if v1 != v2:
                break
            elif retry == 10:
                print('failed to attempt to sample v2 different from v1')
                sys.exit()
            retry += 1

        print(f'v1 {v1}')
        print(f'v2 {v2}')

        first_min = (v2, min_opinion, min_pol)

        # Minimizer start with greedy play
        min_touched.append(v2)

        # store minimizer play history
        min_history.append((v2, min_opinion))
        print(f'min_history: {min_history}')

        # a dictionary {'min_option': count of this choice}
        counter = collections.Counter(min_history)
        print(counter)

        # frequency of all min options in order
        fla_min_fre = np.array(list(counter.values()))/1

        mixed_pol, payoff_row = mixed_min_polarization(
            s, n, A, L, v2, min_opinion, fla_max_fre)
        payoff_matrix = np.vstack([payoff_matrix, payoff_row])
        print('fla_min_fre at the spot')

        min_counter = dict(counter)
        print(f'min_counter: {min_counter}')
        print(min_counter[(v2, min_opinion)])

        equi_min = min_pol
        equi_max = max_pol

        i = 1
        while True:
            if i > game_rounds:
                print(f'min_recent_{memory}_touched: {min_touched}')
                print(f'max_recent_{memory}_touched: {max_touched}')
                print(f'Min last 100 action: {min_touched_last_100}')
                break

            print("_____________________")
            print(f"Game {i}")
            print("_____________________")

            # maximizer play

        
            if i == game_rounds-100:
        
                # max_touched_100 = max_touched
                # min_touched_100 = min_touched
                # max_fre_100 = max_frequency  # store the max_frequency of first 100 iterations
                # print(f'max_history {max_history}')
                # min_fre_100 = fla_min_fre  # max_frequency of first 100 iterations
                # print(f'min_history {min_history}')
                # Remove max frequency less than 0.1--
                max_history_last_100 = np.zeros([n, 2])
                min_history_last_100 = []
                min_touched_last_100 = []

            (v1, max_opinion, equi_max) = mixed_max_play(
                payoff_matrix, s, n, A, L, v2, min_opinion, fla_min_fre)
            max_touched = push(max_touched, v1, memory)

            # cumulate strategy
            max_history[v1, int(max_opinion)] += 1
            max_history_last_100[v1, int(max_opinion)] += 1
            max_frequency = max_history/(i+1)  # its frequency

            # flatten max_frequency to calculate average payoff
            fla_max_fre = max_frequency.flatten()

            # create payoff matrix for maximizer
            row = int(row_index(v2, min_opinion))
            column = int(column_index(v1, max_opinion))

            # _________________________________________________________________
            #         ######################Visualize Maximizer's selection
            # La = scipy.sparse.csgraph.laplacian(G, normed=False)

            # nxG = nx.from_numpy_matrix(G)

            # color_map = []
            # for node in nxG:
            #     if node == v1:
            #         color_map.append('Red')
            #     else:
            #         color_map.append('Grey')

            # #nxG1 = nx.DiGraph(G)
            # nx.draw(nxG, node_color=color_map, with_labels=True,node_size = 50)
            # plt.figure(figsize=(200, 200))
            # plt.show()

            # minimizer play
            (v2, payoff_row, min_opinion, equi_min) = mixed_min_play(
                s, v1, max_opinion, n, A, L, fla_max_fre)
            min_touched = push(min_touched, v2, memory)
            min_touched_all.append(v2)
            min_touched_last_100.append(v2)

            if (v2, round(min_opinion, 2)) in counter.keys():
                # if this min_option is in min_history, no need to update payoff matrix, only update frequency
                payoff_matrix = payoff_matrix
            else:
                # if this is a new option, append to previous matrix
                payoff_matrix = np.vstack([payoff_matrix, payoff_row])

            min_history.append((v2, round(min_opinion, 2)))
            min_history_last_100.append((v2, round(min_opinion)))
            # dictionary {'min_option': count of this choice}
            counter = collections.Counter(min_history)
            # frequency of all min options in order
            fla_min_fre = np.array(list(counter.values()))/(i+1)

            # min_counter = dict(counter)
            # print(min_counter[(v2,min_opinion)]/(i+1)) #get the value from dictionary by using key (v2,opinion)

            # create payoff matrix for minimizer
            row = row_index(v2, min_opinion)
            column = column_index(v1, max_opinion)

            print(
                f"Not Reached Nash Equilibrium at Equi_Min = {equi_min} and Equi_Max = {equi_max}")

            # Visualize Minimizer selection
            # La = scipy.sparse.csgraph.laplacian(G1, normed=False)

            # nxG = nx.from_numpy_array(G1)

            # color_map = []
            # for node in nxG:
            #     if node == v2:
            #         color_map.append('Blue')
            #     else:
            #         color_map.append('Grey')

            # nxG1 = nx.DiGraph(G)
            # nx.draw(nxG, node_color=color_map, with_labels=True)
            # plt.figure(figsize=(25, 25))
            # plt.show()

            if equi_min == equi_max:
                print(
                    f"Reached Nash Equilibrium at game {i} and Equi_Por = {equi_min}")
                print(f'max_distribution:\t{max_frequency}')
                print(f'min_distribution:\t{fla_min_fre}')
                break

            i += 1

        return GameResult(
            first_max=first_max,
            first_min=first_min,
            min_touched=min_touched,
            max_touched=max_touched,
            payoff_matrix=payoff_matrix,
            min_touched_last_100=min_touched_last_100,
            min_touched_all=min_touched_all,
            fla_min_fre=fla_min_fre,
            fla_max_fre=fla_max_fre,
            min_history=min_history,
            max_history=max_history,
            min_history_last_100=min_history_last_100,
            max_history_last_100=max_history_last_100,
        )
