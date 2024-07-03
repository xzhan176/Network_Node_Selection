from math import comb
import sys
import random
import copy
import collections
import numpy as np

# add parent directory to path so that utils is available
sys.path.append('..')

from utils import *


class GameResult:
    """
    Result of K Node Game
    """

    def __init__(self, game_rounds, first_max, first_min, min_touched, max_touched,  payoff_matrix, min_touched_last_100, min_touched_all, fla_min_fre, fla_max_fre, min_history, max_history, min_history_last_100, max_history_last_100, equi_max, equi_min):
        self.game_rounds = game_rounds
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
        self.equi_max = equi_max
        self.equi_min = equi_min


class Game:
    """
    K Node Game
    """

    def __init__(self, s, A, L, k: int):
        self.s = s
        self.n = len(s[:])
        self.A = A
        self.L = L
        self.k = k
        self.h = self._len_actions(k, self.n)
        pass

    ## PRIVATE METHODS ##

    def _len_actions(self, k, n):
        """
        Calculate the length of all possible actions for k opinions and n nodes
        All possible actions = all combination of k nodes (unordered) * all combination of k opinions (ordered)
        """
        # create all combination of k opinions
        max_option = [0, 1]
        k_opinions = list(product(max_option, repeat=k))
        # number of combinations exist
        len_kops = len(k_opinions)
        # Horizontal length of all possible actions
        h = comb(n, k) * len_kops
        return h

    def _cgen(self, i, n, k):
        """
        returns the i-th combination of k numbers chosen from 1,2,...,n
        """
        c = []
        r = i+0
        j = -1
        for s in range(1, k+1):
            cs = j+1
            while r-comb(n-1-cs, k-s) >= 0:
                r -= comb(n-1-cs, k-s)
                cs += 1
            c.append(cs)
            j = cs
        return c

    def _find_idx(self, k_nodes):
        latter = 0
        index = 0
        k = len(k_nodes)

        for i in range(k):
            before = k_nodes[i] + 1
            L_min = latter + 1
            L_max = before - 1

            M = L_max - L_min

            for m in range(1, M+2):
                P = self.n - latter - m
                L = k - 1 - i
                index = index + comb(P, L)
            latter = before

        return index

    def _k_derivate_s(self, v2, M):
        '''
        Derivate min_opinions - using above result
        '''
        # v2 - selection list(k nodes) of minimizer
        #  take the node index from selection list
        #  it's also the column index for these two nodes

        # create a parameter array with all 1/n
        c = np.array([1/self.n] * self.n)
        # c = np.reshape(c, (n,1))

        # Create left side of '=' matrix
        def leftFunction(x, y):
            a_i = np.transpose(self.A[x]-c)@(self.A[y]-c)
            return [a_i]
        a = np.concatenate([leftFunction(x, y) for y in v2 for x in v2])
        a = np.reshape(a, (self.k, self.k))

        # Create right side of '=' matrix
        def rightFunction(x, M):
            Mi = np.dot(M, (self.A[x]-c))
            return -Mi
        b = np.concatenate([rightFunction(x, M) for x in v2])
        result = np.linalg.solve(a, b)
        return result

    def _push(self, obj, element, memory):
        if len(obj) >= memory:
            dif = len(obj) - memory
            obj.pop(dif)
        obj.extend(list(element))
        obj = list(set(obj))
        return obj

    def _convert_available(self, k_nodes, touched):
        touched.sort()
        for i in touched:
            for j in range(self.k):
                if k_nodes[j] >= i:
                    k_nodes[j] = k_nodes[j] + 1
        return k_nodes

    def _create_all_comb(self):
        '''
        create all combination of k opinions
        '''
        max_option = [0, 1]
        return list(product(max_option, repeat=self.k))

    def _create_available_comb(self, i_th, touched):
        '''
        create available combination of K nodes
        '''
        # number of unique touched nodes
        a = len(set(touched))
        # generate the i-th list from total n-a agents
        k_fake = self._cgen(i_th, self.n - a, self.k)
        # convert the i-th list to real k nodes
        k_nodes = self._convert_available(k_fake, touched)
        return k_nodes

    def _change_k_innate_opinion(self, op, node_set, k_opinion):
        '''
        node_set - 1 set  k_opinion- 1 set
        '''
        op = copy.copy(op)

        for j in range(self.k):
            b = node_set[j]  # agent index
            op[b] = k_opinion[j]   # j is index of which opinion combination

        return op

    def _sum_rest(self, op, v2):
        '''
        Create the sum_term - exclude selected nodes
        '''
        # Reshape opinion array
        op = np.reshape(op, (self.n, 1))

        E_new = np.array([1/self.n] * self.n * self.n)
        # create a n*n matrix with all elements 1
        E_new = np.reshape(E_new, (self.n, self.n))
        # A_new = np.reshape(A, (n,n))
        A_new = copy.copy(self.A)
        A_temp = A_new-E_new
        M_new_temp = A_temp@op

        # np.sum(self.A, axis=0)
        # Out_term = np.sum([(lambda x: op[x]*A_temp[x])(x) for x in v2], axis=0)
        def sumFunction(x):
            s_i = op[x]*A_temp[x]
            return s_i
        np.sum(self.A, axis=0)
        Out_term = np.sum([sumFunction(x) for x in v2], axis=0)
        Out_term = np.reshape(Out_term, (self.n, 1))
        M_rest = np.transpose(M_new_temp-Out_term)
        print('M')
        print(M_rest)
        return M_rest

    def _make_k_payoff_row(self, op1, v2):  # op1 here is only changed by Min
        payoff_row = np.zeros(self.h)
        # (node_sets, k_opinions) = self._create_all_comb()

        column = 0
        i = 0

        # i - which set of nodes option
        for i in range(0, comb(self.n, self.k)):
            nodes = self._cgen(i, self.n, self.k)
            k_opinions = self._create_all_comb()

            # tuple index - select one combination of opinions
            for ops in k_opinions:
                # op2 has been changed by both min and max now
                op2 = self._change_k_innate_opinion(op1, nodes, ops)
                check = any(node in nodes for node in v2)

                # when v1 == v2, the polarization should be negative for max, infinite for min.
                # Replace the the column_index of agent v2 with 0 for max
                if check is False:   # if v1 != v2
                    # calculate the payoff polarization
                    payoff = obj_polarization(self.A, op2, self.n)
                    payoff_row[column] = payoff
                    column = column + 1
                else:                # if v1 == v2
                    # use to avoid min and max choose the same agent
                    payoff_row[column] = 10000
                    column = column + 1

        return payoff_row

    def _k_max_polarization(self, payoff_matrix, column, fla_min_fre):
        '''
        Op has been updated by minimizer, fla_min_fre includes min's history, so maximizer react to the innate op after that
        '''
        # create payoff matrix for maximizer
        payoff_vector = payoff_matrix[:, column]
        if any(i > 10 for i in payoff_vector) > 10:
            print('Error in Payoff Matrix')
            sys.exit
        # calculate fictitious payoff - equi_max
        payoff_cal = payoff_vector * fla_min_fre  # payoff * frequency
        mixed_pol = np.sum(payoff_cal)  # add up
        return mixed_pol

    def _k_random_play(self):
        '''
        Player randomly choose an agent and randomly change the agent
        '''
        k_opinions = self._create_all_comb()
        len_nodesets = comb(self.n, self.k)
        # randomly select an agent index
        i_th = random.randint(0, len_nodesets-1)
        v_list = self._cgen(i_th, self.n, self.k)

        # create all combination of K opinions
        len_kops = len(k_opinions)  # - number of combinations exist
        # randomly select index for an OPINION list
        op_index = random.randint(0, len_kops-1)
        # randomly select an opinion list(0 and 1) to update the opinion array
        new_op = k_opinions[op_index]
        print('Nodes, opinions')
        print(v_list, new_op)
        op = self._change_k_innate_opinion(self.s, v_list, new_op)
        por = obj_polarization(self.A, op, self.n)
        column = len_kops*i_th + op_index
        return (v_list, new_op, por, column)

    def _k_random_play_1(self, touched):
        '''
        Player randomly choose an agent and randomly change the agent.
        Will not choose the agent that has been touched by the maximizer.
        '''
        op = copy.copy(self.s)
        # max_opi_option = random.uniform(0, 1)   # options that maximizer have

        # number of unique touched nodes
        a = len(set(touched))
        # number of available combination of k nodes
        len_nodesets = comb(self.n-a, self.k)

        # randomly select an action index
        i_th = random.randint(0, len_nodesets-1)
        v_list = self._create_available_comb(i_th, touched)

        new_op_list = []
        for i in range(self.k):
            new_op = 0.5
            new_op_list.append(new_op)

        new_op_list = tuple(new_op_list)
        print('Nodes, opinions')
        print(v_list, new_op_list)
        op = self._change_k_innate_opinion(self.s, v_list, new_op_list)
        por = obj_polarization(self.A, op, self.n)
        # print(f"Network reaches steady-state Polarization: {por}")
        return (v_list, new_op_list, por)

    def _mixed_K_min_polarization(self, v2, k_opinion, fla_max_fre):
        '''
        Calculate polarization of minimizer's Mixed Strategy
        '''
        # only updated by minimizer's current change
        op1 = self._change_k_innate_opinion(self.s, v2, k_opinion)
        # calculate the polarization with both min(did above) and max's action(in make_payoff_row)
        # the vector list out 2*n payoffs after min's action combine with 2*n possible max's actions
        payoff_row = self._make_k_payoff_row(op1, v2)

        # calculate fictitious payoff - equi_min
        # fla_max_fre recorded the frequency of each maximizer's action, frequency sum = 1
        payoff_cal = payoff_row * fla_max_fre
        # payoff (2*n array) * maximizer_action_frequency (2*n array)

        # add up all, calculate average/expected payoff
        mixed_pol = np.sum(payoff_cal)

        # Replace the the column_index of agent v2 with -100 for max

        payoff_row = [-10000 if ele == 10000 else ele for ele in payoff_row]

        return (mixed_pol, payoff_row)

    def _max_k_play(self, payoff_matrix, fla_min_fre, min_touched):
        # # pass on the innate opinion that has been changed by minimizer
        k_opinions = self._create_all_comb()
        len_kops = len(k_opinions)

        # start producing changes
        all_por = np.zeros(self.h)

        print('fla_min_fre', fla_min_fre[np.nonzero(fla_min_fre)])

        # number of unique touched agent
        a = len(set(min_touched))
        # length of available k_nodes combinations
        len_avsets = comb(self.n - a, self.k)

        for i_th in range(len_avsets):  # for each available k nodes
            v1 = self._create_available_comb(i_th, min_touched)
            # map this node set to its index located in all lists
            k_nodes_index = self._find_idx(v1)

            for f in range(len_kops):         # for each opinion combination
                # locate the column in payoff row- all combinations
                column = k_nodes_index*len_kops + f
                # calculate mixed polarization
                por = self._k_max_polarization(
                    payoff_matrix, column, fla_min_fre)
                all_por[column] = por

            print('Max_por', max(all_por))

        print('all_por', all_por)
        # Index of maximum polarization - in all actions
        column = np.argmax(all_por)
        print(f'column - best action: {column}')

        v1, max_opinion = self.map_action(column)

        print(
            f"Maximizer found its target {self.k} agent: {v1} op: {max_opinion}")

        return v1, max_opinion, np.max(all_por), column

    def _mixed_choose_min_vertex(self, max_touched, fla_max_fre):
        '''
        Minimizer search: Find the best minimizer's action after going through every new_op option of every agent
        '''
        # min_por- set a standard to compare with pol after min's action
        # min_por = obj_polarization(self.A, op, self.n)
        # initial reference value of 1000
        min_por = 1000
        champion = (None, None, 0, None)  # assume the best action is champion

        a = len(set(max_touched))
        len_nodesets = comb(self.n - a, self.k)

        for i_th in range(len_nodesets):
            v2 = self._create_available_comb(i_th, max_touched)
            # find the best new_op option
            changed_opinion, payoff_row, por = self._min_k_mixed_opinion(
                v2, fla_max_fre)

            # if the recent polarization is smaller than the minimum polarization in the history
            if por < min_por:
                # store innate max updated polarization
                min_por = por
                # update the recent option as champion
                champion = (v2, changed_opinion[:], payoff_row, min_por)
                print("champion: ", champion[0], champion[1], champion[3])
            # else:
            #     print('Innate polarization is smaller than Min action')

        return champion

    def _mixed_min_play(self, max_touched, fla_max_fre):
        '''
        Opinions has been updated by maximizer, fla_max_fre includes max's history, so minimizer react to the innate op after that
        '''
        min_champion = self._mixed_choose_min_vertex(max_touched, fla_max_fre)
        (v2, min_opinion, payoff_row, min_pol) = min_champion

        # if minimizer cannot find a action to minimize polarization after maximizer's action
        if v2 == None:
            print('Minimizer fail')

        else:
            print(f"Minimizer finds its target agents: {v2}")

            # Store innate_op of the min_selected k vertex
            # old_opinion_min = [self.s[i] for i in v2]
            # print(f"    Agent {v2} 's opinion {old_opinion_min} changed to {min_opinion}")

            # print("Network reaches steady-state Polarization: {min_pol}")

        return (tuple(v2), payoff_row, min_opinion, min_pol)

    def _min_k_mixed_opinion(self, v2, fla_max_fre):
        weight_M = 0
        # loop for each max_action(in total 2*n)
        k_opinions = self._create_all_comb()
        len_kops = len(k_opinions)

        for column in range(self.h):
            if fla_max_fre[column] != 0:
                if column > self.k:
                    nodeset_index = int(column/len_kops)
                    opset_index = column % len_kops
                else:
                    # print('less than 1')
                    nodeset_index = 0
                    opset_index = column

                # Calculating Max's action at this column
                v1 = self._cgen(nodeset_index, self.n, self.k)
                max_opinion = k_opinions[opset_index]

                # change innate opinion by max action
                op1 = self._change_k_innate_opinion(self.s, v1, max_opinion)

                # Derivate optimal Min's opinion for nodeset v2
                # {sum}{j}(s_j(h_j -c))  - rest of terms
                M_rest = self._sum_rest(op1, v2)

                weight_M += fla_max_fre[column]*M_rest  # {sum}{v} p_v * M

        # Got optimal Min's opinion for v2
        # give a set of k weighted opinions
        k_opinion = self._k_derivate_s(v2, weight_M)
        updated_k_opinion = [0 if x < 0
                             else 1 if x > 1
                             else x
                             for x in k_opinion]

        (mixed_por, payoff_row) = self._mixed_K_min_polarization(
            v2, updated_k_opinion, fla_max_fre)

        return (updated_k_opinion, payoff_row, mixed_por)

    ## PUBLIC METHODS ##

    def setK(self, k):
        self.k = k

    def map_action(self, column):
        k_opinions = self._create_all_comb()
        len_kops = len(k_opinions)
        nodeset_index = int(column / len_kops)
        opset_index = column % len_kops
        k_nodes = self._cgen(nodeset_index, self.n, self.k)
        opinions = k_opinions[opset_index]
        return (k_nodes, opinions)

    def run(self, game_rounds: int, memory: int):
        payoff_matrix = np.empty((0, self.h), float)
        max_touched = []
        min_touched = []
        min_touched_all = []
        min_touched_last_100 = []
        # n*2 matrix, agent i & opinion options
        max_history = np.zeros(self.h, int)
        min_history = []
        max_history_last_100 = np.zeros(self.h, int)
        min_history_last_100 = []

        # Game start from maximizer random play
        (v1, max_opinion, max_pol, column) = self._k_random_play()

        # Save the first action to report later
        first_max = (v1, max_opinion, max_pol)

        # save Maximizer's action history
        max_touched.extend(tuple(v1))

        # store maximizer play history, using agent(row) and changed opinion(column) as indicator to locate history
        max_history[column] += 1

        # its frequency, only played 1 time so far, divided by 1
        fla_max_fre = max_history / 1

        # if game start from minimizer random play - make sure two random play are not same agent!!!
        # print('Minimizer first selection')
        (v2, min_opinion, min_pol) = self._k_random_play_1(v1)

        first_min = (v2, min_opinion, min_pol)

        min_touched.extend(v2)
        min_touched_all.append(tuple(v2))

        # store minimizer play history
        # min_history.append((v2 + tuple(min_opinion)))
        min_history.append((tuple(v2), tuple(min_opinion)))
        print(f'min_history {min_history}')

        # a dictionary include {'min_option': count of this choice}
        counter = collections.Counter(min_history)

        # frequency of all min options in order
        fla_min_fre = np.array(list(counter.values()))/1

        (_, payoff_row) = self._mixed_K_min_polarization(
            v2, min_opinion, fla_max_fre)
        payoff_matrix = np.vstack([payoff_matrix, payoff_row])

        # min_counter = dict(counter)
        # print(min_counter[(v2,min_opinion)]/(i+1)) #get the value from dictionary by using key (v2,opinion)
        equi_min = min_pol
        equi_max = max_pol

        i = 1
        while True:
            # has finished all game rounds
            if i > game_rounds:
                print('MAX_last_100,  all')
                max_l100_fre = max_history_last_100/100
                max_fre = max_history/game_rounds
                print(max_l100_fre[np.nonzero(max_l100_fre)],
                      max_fre[np.nonzero(max_fre)])
                print(np.nonzero(max_l100_fre)[0],
                      np.nonzero(max_fre)[0])

                columns = list(np.nonzero(max_l100_fre)[0])
                for column in list(columns):
                    k_opinions = self._create_all_comb()
                    len_kops = len(k_opinions)
                    nodeset_index = int(column/len_kops)
                    opset_index = column % len_kops
                    k_nodes = self._cgen(nodeset_index, self.n, self.k)
                    opinions = k_opinions[opset_index]
                    print(f'Max Nodes: {k_nodes} | Opinion: {opinions}')

                # MINimizer's Strategy in the last 100 round
                counter = collections.Counter(min_touched_last_100)
                # frequency of all min options in order
                fla_min_fre = np.array(list(counter.values()))/(100)
                print('MIN_last_100,  all')
                # a dictionary {'min_option': count of this choice}
                counter_1 = collections.Counter(min_touched_all)
                # frequency of all min options in order
                fla_min_fre_1 = np.array(list(counter_1.values()))/game_rounds
                print(fla_min_fre, fla_min_fre_1)
                print(counter, counter_1)
                print(f'Max Pol: {equi_max} Min Pol: {equi_min}')
                break


            print("-" * 20)
            print(f"Game {i}")
            print("-" * 10)

            print(f'min_history {min_history}')
            print(f'max_history {max_history}')


            if i == game_rounds - 100:
                # Remove max frequency less than 0.1--
                max_history_last_100 = np.zeros(self.h)
                min_history_last_100 = []
                min_touched_last_100 = []

            # MAXimizer play
            (v1, max_opinion, equi_max, column) = self._max_k_play(
                payoff_matrix, fla_min_fre, min_touched)
            max_touched = self._push(max_touched, v1, memory)

            # cumulate strategy
            max_history[column] += 1
            max_history_last_100[column] += 1

            # max_frequency to calculate average payoff
            fla_max_fre = max_history/(i+1)
            print(f'fre_max at spot {fla_max_fre[column]}')

            # MINimizer play
            (v2, payoff_row, min_opinion, equi_min) = self._mixed_min_play(
                max_touched, fla_max_fre)
            min_touched = self._push(min_touched, v2, memory)
            min_touched_all.append(v2)
            min_touched_last_100.append(v2)

            if tuple(tuple(v2) + min_opinion) in counter.keys():
                # if (v2, tuple(min_opinion)) in counter.keys():
                # if this min_option is in min_history, no need to update payoff matrix, only update frequency
                payoff_matrix = payoff_matrix
            else:
                # if this is a new option, append to previous matrix
                payoff_matrix = np.vstack([payoff_matrix, payoff_row])

            # min_history.append((v2 + tuple(min_opinion)))
            min_history.append((v2, tuple(min_opinion)))
            min_history_last_100.append((v2, min_opinion))

            # a dictionary include {'min_option': count of this choice}
            counter = collections.Counter(min_history)
            # frequency of all min options in order
            fla_min_fre = np.array(list(counter.values()))/(i+1)
            print(f'fla_min_fre: {fla_min_fre}')

            if equi_min == equi_max:
                print(
                    f"Reached Nash Equilibrium at game {i} and Equi_Por = {equi_min}")
                # print(f'max_distribution {max_frequency}')
                # print(f'min_distribution {fla_min_fre}')
                break
            else:
                print(
                    f"Not Reached Nash Equilibrium at Equi_Min = {equi_min} and Equi_Max = {equi_max}")

            i += 1

        return GameResult(
            game_rounds=game_rounds,
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
            equi_max=equi_max,
            equi_min=equi_min,
        )
