import numpy as np
import collections
import sys
import copy
import random


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
    def __init__(self, s, A, L, calculate_polarization):
        self.s = s
        self.n = len(s)
        self.A = A
        self.L = L
        self.calculate_polarization = calculate_polarization

    # PRIVATE METHODS

    def _push(self, objs, element, memory):
        if len(objs) >= memory:
            objs.pop(0)
            print('pop')
        objs.append(element)
        return objs

    def _row_index(self, v2, min_opinion):
        row = 11*v2 + min_opinion*10
        return int(row)

    def _column_index(self, v1, max_opinion):
        """
        Returns data frame index
        """
        column = 2*v1 + max_opinion
        return int(column)

    def _random_play(self):
        """
        Player randomly chooses an agent and randomly change the agent's opinion
        """
        op = copy.copy(self.s)
        # randomly select an agent index
        v_index = random.randint(0, self.n - 1)
        # randomly select an opinion either 0 and 1
        new_opinion = random.uniform(0, 1)
        # print(f'new_op: {new_opinion}')

        # Store old opinion
        old_opinion = op[v_index, 0]

        # update the opinion
        op[v_index, 0] = new_opinion
        print(
            f"    Agent{v_index}'s opinion {old_opinion} changed to {new_opinion}")
        polar = self.calculate_polarization(self.A, op, self.n)

        # restore op array to innate opinion
        op[v_index] = old_opinion
        print(f"Network reaches equilibrium Polarization: {polar}")

        return v_index, new_opinion, polar

    def _make_payoff_row(self, op1, v2):
        payoff_row = np.zeros(2 * self.n)

        for column in range(2 * self.n):
            v1 = int(column/2)  # i.e., column 11 is agent 5, opinion 1
            max_opinion = column % 2
            # update the maximizer's change to the opinion array that has changed by minimizer(op1)
            op2 = copy.copy(op1)
            op2[v1, 0] = max_opinion

            # calculate the polarization with both max and min's action
            payoff_row[column] = self.calculate_polarization(
                self.A, op2, self.n)

        # when v1 == v2, the polarization should be negative for max, infinite for min.
        # ZERO SUM when v1==v2, the polarization is innate polarization
        j_1 = 2*v2 + 0
        j_2 = 2*v2 + 1
        O_P = self.calculate_polarization(self.A, self.s, self.n)
        payoff_row[j_1] = O_P
        payoff_row[j_2] = O_P

        return payoff_row

    def _derivate_s(self, op, v2):
        """
        Parameters:
        - op: opinion array that updated by maximizer
        """
        c = [1 / self.n] * self.n
        sum_term = 0
        j = 0

        # sum up all terms
        sum_term = np.dot(np.dot((self.A-c), (self.A[v2]-c)), op)

        # exclude the term that j = v2
        term_out = op[v2] * np.dot((self.A[v2]-c), (self.A[v2]-c))
        sum_s = sum_term - term_out    # numerator

        s_star = -sum_s/np.dot((self.A[v2]-c), (self.A[v2]-c))
        s_star = s_star[0]  # take value out of array
        min_opinion = min(max(0, s_star), 1)

        return min_opinion

    def _min_mixed_opinion_1(self, v2, fla_max_fre):
        weight_op = 0

        # loop for each max_action(in total 2*n)
        for column in range(2 * self.n):

            if fla_max_fre[column] != 0:
                v1 = int(column/2)  # i.e., column 11 is agent 5, opinion 1
                max_opinion = column % 2
                op = copy.copy(self.s)
                op[v1] = max_opinion

                # find min_s_star for each max_action
                min_opinion = self._derivate_s(op, v2)
                op1 = copy.copy(op)
                # after max action, update min action on opinion array
                op1[v2] = min_opinion
                # print(min_opinion)
                # min_por = self.calculate_polarization(self.A, op1, self.n)
                # t = 0
                weight_op += fla_max_fre[column]*min_opinion  # sum up p_i*s_i

        mixed_por, payoff_row = self._mixed_min_polarization(
            v2, weight_op, fla_max_fre)

        return weight_op, payoff_row, mixed_por

    def _mixed_min_polarization(self, v2, weight_op, fla_max_fre):
        """
        Calculate polarization of minimizer's Mixed Strategy
        """
        op1 = copy.copy(self.s)
        op1[v2, 0] = weight_op  # update by minimizer's current change

        # calculate the polarization with both min(did here) and max's action(in make_payoff_row)
        # the vector list out 2*n payoffs after min's action combine with 2*n possible max's actions
        payoff_row = self._make_payoff_row(op1, v2)

        # calculate fictitious payoff - equi_min
        # fla_max_fre recorded the frequency of each maximizer's action, frequency sum = 1
        payoff_cal = payoff_row * fla_max_fre
        # payoff (2*n array) * maximizer_action_frequency (2*n array)

        # add up all, calculate average/expected payoff
        mixed_pol = np.sum(payoff_cal)

        return (mixed_pol, payoff_row)

    def _mixed_choose_min_vertex(self, v1, max_opinion, fla_max_fre):
        """
        Find the best minimizer's action after going through every agent's option
        """
        # current polarization that changed by maximizer, "innate" objective that min start with
        op = copy.copy(self.s)
        op[v1, 0] = max_opinion

        min_por = 1000  # use the infinite big min_por
        champion = (None, None, 0, None)  # assume the best action is champion

        for v2 in range(self.n):
            ################################# for ZERO SUM ##########################################
            if v2 == v1:
                por, payoff_row = self._mixed_min_polarization(
                    v2, self.s[v2], fla_max_fre)
                # doesn't change the innate opinion, keep the polarization as innate polarization
                changed_opinion = self.s[v2, 0]
            else:
                # find the best new_op option
                changed_opinion, payoff_row, por = self._min_mixed_opinion_1(
                    v2, fla_max_fre)

            # the recent polarization is smaller than the minimum polarization in the history
            if por < min_por:
                min_por = por
                # update the recent option as champion
                champion = (v2, changed_opinion, payoff_row, min_por)
            # else:
                # print('Innate polarization is smaller than Min action')

        return champion

    def _mixed_min_play(self, v1, max_opinion, fla_max_fre):
        """
        Op has been updated by maximizer, fla_max_fre includes max's history,
        so minimizer react to the innate op after that
        """

        min_champion = self._mixed_choose_min_vertex(
            v1, max_opinion, fla_max_fre)
        v2, min_opinion, payoff_row, min_pol = min_champion

        # minimizer cannot find a action to minimize polarization after maximizer's action
        if v2 == None:
            print('Minimizer fail')

        else:
            print("Minimizer found its target agent")

            # Store innate_op of the min_selected vertex
            # old_opinion_min = op[v2, 0]
            old_opinion_min = self.s[v2, 0]

            print(
                f"    Agent{v2}'s opinion {old_opinion_min} changed to {min_opinion}")

        return (v2, payoff_row, min_opinion, min_pol)

    def _mixed_max_polarization(self, payoff_matrix, v1, max_opinion, fla_min_fre):
        """
        Op has been updated by minimizer, fla_min_fre includes min's history,
        so maximizer react to the innate op after that
        """
        # create payoff matrix for maximizer
        column = int(self._column_index(v1, max_opinion))
        payoff_vector = payoff_matrix[:, column]

        # calculate fictitious payoff - equi_max
        payoff_cal = payoff_vector * fla_min_fre  # payoff * frequency

        mixed_pol = np.sum(payoff_cal)

        return mixed_pol

    def _max_mixed_opinion(self, payoff_matrix, v1, fla_min_fre):
        """
        Determines if value of opinion at v should be set to 0 or 1 to maximize equilibrium polarization
        """
        # create a two_element array to store polarization value of each option
        por_arr = np.zeros(2)

        # Maximizer has two options to change agent v1's opinion
        max_opi_option = [0, 1.0]

        # objective if set opinion to 0, 1.0
        j = 0
        for new_op in max_opi_option:
            # print(f'change op to {str(i/10)}')
            max_opinion = new_op
            por_arr[j] = self._mixed_max_polarization(
                payoff_matrix, v1, max_opinion, fla_min_fre)
            j += 1   # index increase 1, put the polarization in array

        # the index of maximum polarization = max_opinion --[0,1]
        maximize_op = np.argmax(por_arr)
        # find the maximum polarization in the record
        max_por = np.max(por_arr)

        return maximize_op, max_por

    def _mixed_choose_max_vertex(self, payoff_matrix, v2, fla_min_fre):
        """
        Determine which agent maximizer should select to maximizer the equilibrium polarization
        """
        # use "innate"(after min action) polarization as a comparable standard to find max_por
        max_por = 0

        champion = (None, None, max_por)  # assume champion is the best action
        for v1 in range(self.n):
            changed_opinion, por = self._max_mixed_opinion(
                payoff_matrix, v1, fla_min_fre)

            if v2 == v1:
                # doesn't change the innate opinion, keep the polarization as innate polarization
                changed_opinion = self.s[v2, 0]

            # the polarization of most recent action > maximum polarization of previous actions
            if por > max_por:
                max_por = por
                champion = (v1, changed_opinion, max_por)
            # else:
                # print('Innate polarization is bigger than max action')

        return champion

    def _mixed_max_play(self, payoff_matrix, v2, min_opinion, fla_min_fre):
        """
        Parameters:
        - s: innate opinion
        """
        op = copy.copy(self.s)

        # update innate opinion
        # Op has been updated by minimizer, so maximizer react to the innate op after that
        op[v2, 0] = min_opinion

        # The best choice among all opinions and vertices
        max_champion = self._mixed_choose_max_vertex(
            payoff_matrix, v2, fla_min_fre)
        v1, max_opinion, max_pol = max_champion

        if v1 == None:
            print('Maximizer fail')

        else:
            print("Maximizer finds its target agent")
            # Store innate_op of the max_selected vertex
            old_opinion_max = op[v1, 0]

        if v1 == v2:
            # If select the same agent, doesn't change the opinion
            max_opinion = self.s[v1, 0]

            # check if agent's opinion is changed or not
        print(
            f"    Agent{v1}'s opinion {old_opinion_max} changed to {max_opinion}")

        return v1, max_opinion, max_pol

    # PUBLIC METHODS

    def run(self, game_rounds, memory):
        payoff_matrix = np.empty((0, 2 * self.n), float)
        max_touched = []
        min_touched = []
        min_touched_all = []
        min_touched_last_100 = []
        # n*2 matrix, agent i & opinion options
        max_history = np.zeros([self.n, 2])
        min_history = []
        max_history_last_100 = np.zeros([self.n, 2])
        min_history_last_100 = []

        # Game start from maximizer random play
        # Maximizer start with random action
        v1, max_opinion, max_pol = self._random_play()

        # Save the first action to report later
        first_max = (v1, max_opinion, max_pol)

        # save Maximizer's action history
        max_touched.append(v1)

        # store maximizer play history, using agent(row) and changed opinion(column) as indicator to locate history
        max_history[v1, int(max_opinion)] = + 1

        # its frequency, only played 1 time so far, divided by 1
        max_frequency = max_history / 1

        column = self._column_index(v1, max_opinion)

        # flatten the n*2 matrix to a 2n*1 matrix
        fla_max_fre = max_frequency.flatten()
        print(f'fre_max at spot: {fla_max_fre[column]}')

        # the frequency of maximizer's most recent action (v1,max_opinion)

        # if game start from minimizer random play - make sure two random play are not same agent!!!
        while True:
            retry = 0
            v2, min_opinion, min_pol = self._random_play()
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

        mixed_pol, payoff_row = self._mixed_min_polarization(
            v2, min_opinion, fla_max_fre)
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

            if i == game_rounds - 100:
                # max_touched_100 = max_touched
                # min_touched_100 = min_touched
                # max_fre_100 = max_frequency  # store the max_frequency of first 100 iterations
                # print(f'max_history {max_history}')
                # min_fre_100 = fla_min_fre  # max_frequency of first 100 iterations
                # print(f'min_history {min_history}')
                # Remove max frequency less than 0.1--
                max_history_last_100 = np.zeros([self.n, 2])
                min_history_last_100 = []
                min_touched_last_100 = []

            # maximizer play
            (v1, max_opinion, equi_max) = self._mixed_max_play(
                payoff_matrix, v2, min_opinion, fla_min_fre)
            max_touched = self._push(max_touched, v1, memory)

            # cumulate strategy
            max_history[v1, int(max_opinion)] += 1
            max_history_last_100[v1, int(max_opinion)] += 1
            max_frequency = max_history/(i+1)  # its frequency

            # flatten max_frequency to calculate average payoff
            fla_max_fre = max_frequency.flatten()

            # create payoff matrix for maximizer
            row = int(self._row_index(v2, min_opinion))
            column = int(self._column_index(v1, max_opinion))

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
            (v2, payoff_row, min_opinion, equi_min) = self._mixed_min_play(
                v1, max_opinion, fla_max_fre)
            min_touched = self._push(min_touched, v2, memory)
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
            row = self._row_index(v2, min_opinion)
            column = self._column_index(v1, max_opinion)

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
