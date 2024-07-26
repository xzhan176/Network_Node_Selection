from utils import *
from math import comb
from itertools import combinations, product, islice
from joblib import Parallel, delayed, dump, load
import os
import sys
import random
import collections
import numpy as np

PRECISION = 3
# DISPLAY_BENCHMARK = False
DISPLAY_BENCHMARK = True


class GameResult:
    """
    Result of K Node Game
    """

    def __init__(self, game_rounds: int, first_max, first_min, min_touched, max_touched,  payoff_matrix, min_touched_last_100, min_touched_all, fla_min_fre, fla_max_fre, min_history, max_history, min_history_last_100, max_history_last_100, equi_max, equi_min):
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

    @staticmethod
    def loadDataToDisk(data, filename: str, folder: str = 'memmaps'):
        """
        Use this static method to load s and A to disk for improved performance
        """
        try:
            folderPath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                folder)
            os.mkdir(folderPath)
        except FileExistsError:
            pass

        data_memmap = os.path.join(folderPath, filename)
        dump(data, data_memmap)

    def __init__(self, k: int, s=None, A=None,
                 temp_path: str | None = None,
                 s_memmap_file: str | None = None,
                 A_memmap_file: str | None = None,
                 disk_dumped: bool = False):

        self._temp_path = temp_path or os.path.join(
            os.path.dirname(__file__),
            "temp")
        os.makedirs(self._temp_path, exist_ok=True)

        # Init s, use data from disk if available
        s_memmap_file = s_memmap_file or os.path.join(
            self._temp_path,
            "s_memmap")
        if not disk_dumped:
            dump(s, s_memmap_file)
        self.s = load(s_memmap_file, mmap_mode='r')

        self.n = len(s[:])

        # Make sure k is a valid value
        if k * 2 > self.n:
            raise Exception(
                'Invalid k value. Cannot have k*2 > n (size of network).')

        # Init A, use data from disk if available
        A_memmap_file = A_memmap_file or os.path.join(
            self._temp_path,
            "A_memmap")
        if not disk_dumped:
            dump(A, A_memmap_file)
        self.A = load(A_memmap_file, mmap_mode='r')

        self.k = k
        self.h = len_actions(k, self.n)

    ## PRIVATE METHODS ##

    def _k_derivate_s(self, v2, M):
        '''
        Derivate min_opinions
        '''
        # v2 - selection list(k nodes) of minimizer
        #  take the node index from selection list
        #  it's also the column index for these two nodes

        n = self.n
        A = self.A
        k = self.k

        # parameter array with all 1/n
        c = np.full(n, 1/n)

        # Left side of '=' matrix equation
        def leftFunction(x, y):
            return [np.transpose(A[x]-c)@(A[y]-c)]
        a = np.concatenate([leftFunction(x, y) for y in v2 for x in v2])
        a = np.reshape(a, (k, k))

        # Right side of '=' matrix equation
        def rightFunction(x, M):
            return -np.dot(M, (A[x]-c))
        b = np.concatenate([rightFunction(x, M) for x in v2])
        result = np.linalg.solve(a, b)
        return result

    def _push(self, l: list, k_node: list | tuple, memory: int):
        lSize = len(l)
        if lSize >= memory:
            diff = lSize - memory
            for _ in range(diff):
                l.pop(0)

        l.extend(k_node)
        l = list(set(l))
        return l

    def _next_available_combination(self, i_th: int, touched: list):
        '''
        Return the next available combination of k nodes starting from i_th position
        '''
        # number of unique touched nodes
        a = len(set(touched))
        # generate the i-th list from n-tLen agents
        k_nodes = get_k_node(i_th, self.n - a, self.k)
        # convert the i-th list to real k nodes
        convert_available(self.k, k_nodes, touched)
        return k_nodes

    def _sum_rest(self, op, v2):
        '''
        Return the sum_term, excluding nodes in v2.
        '''
        # Reshape opinion array
        op = np.reshape(op, (self.n, 1))

        E_new = np.array([1/self.n] * self.n * self.n)
        # create a n*n matrix with all elements 1
        E_new = np.reshape(E_new, (self.n, self.n))
        # A_new = np.reshape(A, (n,n))
        A_temp = self.A - E_new
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
        return M_rest

    # [x] improved a bit
    # TODO optimize _make_k_payoff_row
    def _make_k_payoff_row(self, op, v2):
        """
        - op: opinions that was changed by minimizer

        Return the payoff row for the minimizer with shape (h, ) filled with floats
        """
        # make default values infinite then we will override them later. Infinite values will make the minimizer avoid choosing these actions.
        payoffs = np.full(self.h, np.inf)
        column = 0

        # print(f'_make_k_payoff_row: iterate through {comb(self.n, self.k) * 2**self.k} columns') # TODO remove
        # iterate through all possible k nodes combinations excluding v2 nodes
        for k_node in combinations([x for x in range(self.n) if x not in v2], self.k):
            k_opinions, k_opinions_size = max_k_opinion_generator(self.k)
            # s = time.time() # TODO remove
            for k_opinion_index, k_opinion in enumerate(k_opinions):
                column = find_index(self.n, k_node) * \
                    k_opinions_size + k_opinion_index
                # opinions that was changed by both min and max
                op_min_max = change_k_innate_opinion(op, k_node, k_opinion)
                # TODO for zero sum, when v1 == v2, pass self.s instead of op_min
                payoff_polarization = obj_polarization(
                    self.A, op_min_max, self.n)
                # payoff_polarization = fn_benchmark(
                #     lambda: obj_polarization(self.A, op_min_max, self.n),
                #     label=f"_make_k_payoff_row: Calculate polarization for column {column}/{self.h}",
                # )
                payoffs[column] = payoff_polarization
            # e = time.time() # TODO remove
            # et = e - s
            # print(f'TEST benchmark {et:.5f} seconds') # TODO remove

        return payoffs

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
        payoff_cal = payoff_vector * fla_min_fre
        mixed_pol = np.sum(payoff_cal)
        return mixed_pol

    def _k_max_polarization_batch(self, v1s, max_k_opinion_size, payoff_matrix, fla_min_fre):
        # (column index, polarization)
        best = (0, 0)

        for v1 in v1s:
            v1_index = find_index(self.n, v1)
            for f in range(max_k_opinion_size):  # for each opinion combination
                # locate the column in payoff row - all combinations
                column = v1_index * max_k_opinion_size + f
                mixed_polarization = self._k_max_polarization(
                    payoff_matrix, column, fla_min_fre)
                if mixed_polarization > best[1]:
                    best = (column, mixed_polarization)

        return best

    def _k_random_play(self):
        '''
        Player randomly choose an agent and randomly change the agent
        '''
        k_opinions, len_kops = max_k_opinion_generator(self.k)
        len_nodesets = comb(self.n, self.k)
        # randomly select an agent index
        i_th = random.randint(0, len_nodesets-1)
        v_list = get_k_node(i_th, self.n, self.k)

        # randomly select index for an OPINION list
        op_index = random.randint(0, len_kops-1)
        # randomly select an opinion list(0 and 1) to update the opinion array
        new_op = next(islice(k_opinions, op_index, None))
        # print('Nodes, opinions')
        # print(v_list, new_op)
        op = change_k_innate_opinion(self.s, v_list, new_op)
        por = obj_polarization(self.A, op, self.n)
        column = len_kops*i_th + op_index
        return (v_list, new_op, por, column)

    def _k_random_play_1(self, touched: list):
        '''
        Player randomly choose an agent and randomly change the agent.
        Will not choose the agent that has been touched by the maximizer.
        '''
        # number of unique touched nodes
        a = len(set(touched))
        # number of available combination of k nodes
        len_nodesets = comb(self.n-a, self.k)

        # randomly select an action index
        i_th = random.randint(0, len_nodesets-1)
        v_list = self._next_available_combination(i_th, touched)

        new_op_list = []
        for _ in range(self.k):
            new_op = 0.5
            new_op_list.append(new_op)

        # print('Nodes, opinions')
        # print(v_list, new_op_list)
        updated_op = change_k_innate_opinion(self.s, v_list, new_op_list)
        por = obj_polarization(self.A, updated_op, self.n)
        return (v_list, new_op_list, por)

    def _mixed_k_min_polarization(self, v2, k_opinion, fla_max_fre):
        '''
        Calculate polarization of minimizer's Mixed Strategy
        '''
        # only updated by minimizer's current change
        op_min = change_k_innate_opinion(self.s, v2, k_opinion)

        # calculate the polarization with both min(did above) and max's action(in make_payoff_row)
        # the vector list out 2*n payoffs after min's action combine with 2*n possible max's actions
        payoff_row = self._make_k_payoff_row(op_min, v2)
        # payoff_row = fn_benchmark( # TODO remove benchmark
        #     lambda: self._make_k_payoff_row(op_min, v2),
        #     label=f"_make_k_payoff_row: Minimizer find best action for {v2}",
        #     display=DISPLAY_BENCHMARK,
        # )

        # fla_max_fre recorded the frequency of each maximizer's action, frequency sum = 1
        # payoff (h array) * maximizer_action_frequency
        payoff_cal = payoff_row * fla_max_fre

        # add up all, calculate average/expected payoff
        mixed_pol = np.sum(payoff_cal)

        # make default values infinite then we will override them later. Infinite values will make the minimizer avoid choosing these actions.
        payoff_row = [-np.inf if payoff == np.inf
                      else payoff
                      for payoff in payoff_row]

        return (mixed_pol, payoff_row)

    # [x] improved
    def _max_k_play(self, payoff_matrix, fla_min_fre, min_touched: list):
        _, max_k_opinion_size = max_k_opinion_generator(self.k)
        all_por = np.zeros(self.h)

        # TODO optimize this loop
        for v1 in combinations([x for x in range(self.n) if x not in min_touched], self.k):
            v1_index = find_index(self.n, v1)

            for f in range(max_k_opinion_size):  # for each opinion combination
                # locate the column in payoff row - all combinations
                column = v1_index * max_k_opinion_size + f
                mixed_polarization = self._k_max_polarization(
                    payoff_matrix, column, fla_min_fre)
                all_por[column] = mixed_polarization

        #     print('Max_por', max(all_por))

        # print('all_por', all_por)
        # Index of maximum polarization - in all actions
        column = np.argmax(all_por)
        print(f'Max play best action column: {column}')
        max_pol = round(all_por[column], PRECISION)

        v1, v1_opinion = self.map_action(column)

        print(
            f"Maximizer found its target {self.k} agents: {v1} op: {v1_opinion}")

        return v1, v1_opinion, max_pol, column

    # with multiprocessing
    def _max_k_play_multi(self, payoff_matrix, fla_min_fre, min_touched: list):
        # all_por = np.zeros(self.h)
        _, max_k_opinion_size = max_k_opinion_generator(self.k)
        # TODO for zero sum, do not exclude min_touched
        # TODO any function that reference k will need to be updated to the size of min_touched
        k_node_count = comb(self.n - len(min_touched), self.k)
        option_count = k_node_count * max_k_opinion_size

        print(f'Maximizer: iterating through {option_count} options ({
              k_node_count} * {max_k_opinion_size})')  # TODO remove

        cpus = os.cpu_count()

        batch_size = int(1e5)
        data = np.array(
            list(combinations([x for x in range(self.n) if x not in min_touched], self.k)))
        batch_count = max(int(option_count / batch_size), 1)
        jobs = min(batch_count, cpus)
        print(f'Maximizer: batch count {batch_count} - jobs: {jobs}')
        batches = np.array_split(data, batch_count)

        results = Parallel(n_jobs=jobs)(
            delayed(self._k_max_polarization_batch)
            (v1s, max_k_opinion_size, payoff_matrix, fla_min_fre)
            for v1s in batches)

        champion = max(results, key=lambda x: x[1])
        column, max_por = champion
        print(f'Maximizer: champion {champion} (column, max_por)')

        # print('all_por', all_por)

        # get k node corresponding to the column
        v1, max_opinion = self.map_action(column)

        print(
            f"Maximizer found its target {self.k} agents: {v1} op: {max_opinion}")

        return v1, max_opinion, max_por, column

    def _mixed_choose_min_vertex_multi(self, max_touched: list, fla_max_fre):
        '''
        Go through each opinion option of each agent to find the best minimizer's action
        '''
        # TODO for zero sum, do not exclude max_touched
        # TODO any function that reference k will need to be updated to the size of max_touched
        available_nodes = [x for x in range(self.n) if x not in max_touched]
        available_k_nodes_count = comb(len(available_nodes), self.k)
        cpus = os.cpu_count()

        print(
            f'Minimizer: finding the champion of {available_k_nodes_count} available k nodes')

        champion_dtype = np.dtype([
            ('v2', np.int32, (self.k,)),
            ('min_opinion', np.float64, (self.k,)),
            ('payoff_row', np.float64, (self.h)),
            ('min_pol', np.float16),
        ])
        champion_file = os.path.join(self._temp_path, 'min_champion')
        champion = np.memmap(
            champion_file, dtype=champion_dtype, mode='w+', shape=(1, ))
        champion[0]['min_pol'] = 1000

        def find_champion(champion, v2, fla_max_fre):
            candidate = self._min_k_mixed_opinion(v2, fla_max_fre)
            if candidate[3] < champion[0]['min_pol']:
                champion[0] = candidate

        available_k_nodes = combinations(available_nodes, self.k)
        # champion_candidates = Parallel(n_jobs=cpus)(
        # delayed(self._min_k_mixed_opinion)(v2, fla_max_fre)
        # for v2 in available_k_nodes)
        Parallel(n_jobs=cpus)(
            delayed(find_champion)(champion, v2, fla_max_fre)
            for v2 in available_k_nodes)

        champion = tuple(champion[0][0]), tuple(
            champion[0][1]), champion[0][2], champion[0][3]

        print(f'v2 champion {champion}')
        return champion

    def _mixed_choose_min_vertex(self, max_touched: list, fla_max_fre):
        '''
        Go through each opinion option of each agent to find the best minimizer's action
        '''
        # set a standard to compare with pol after min's action
        # initial reference value of 1000
        min_por = 1000
        # best action
        champion = (None, None, 0, None)

        available_k_nodes = [x for x in range(self.n) if x not in max_touched]

        print(
            f'Iterating through {comb(len(available_k_nodes), self.k)} available k nodes')

        # TODO optimize min_por
        for v2 in combinations(available_k_nodes, self.k):
            _, changed_opinion, payoff_row, por = self._min_k_mixed_opinion(
                v2, fla_max_fre)
            # changed_opinion, payoff_row, por = fn_benchmark( # TODO remove benchmark
            #     lambda: self._min_k_mixed_opinion(v2, fla_max_fre),
            #     label=f"_min_k_mixed_opinion: Minimizer find best action for {v2}",
            #     display=DISPLAY_BENCHMARK,
            # )
            # print(f'v2: {v2} por {por}') # TODO remove

            # if the recent polarization is smaller than the minimum polarization in the history
            if por < min_por:
                # store innate max updated polarization
                min_por = por
                # update the recent option as champion
                champion = (v2, tuple(changed_opinion), payoff_row, min_por)
                print(
                    f"v2 champion: {champion[0]} {champion[1]} por: {champion[3]}")
            # else:
            #     print('Innate polarization is smaller than Min action')

        return champion

    def _mixed_min_play(self, max_touched: list, fla_max_fre):
        '''
        Opinions has been updated by maximizer, fla_max_fre includes max's history, so minimizer react to the innate op after that
        '''
        # v2, min_opinion, payoff_row, min_pol = self._mixed_choose_min_vertex(
        v2, min_opinion, payoff_row, min_pol = self._mixed_choose_min_vertex_multi(
            max_touched, fla_max_fre)

        # if minimizer cannot find a action to minimize polarization after maximizer's action
        # if v2 == None:
        if min_pol == 1000:
            print('Minimizer fail')
        else:
            print(f"Minimizer found its target agents: {v2} {min_opinion}")

        return (tuple(v2), min_opinion, payoff_row, min_pol)

    def _min_k_mixed_opinion(self, v2: list, fla_max_fre):
        """
        Return (v2, derivative_k_opinion, payoff_row, mixed_por)
        """
        weight_M = 0

        # print(f'loop through {self.h} max actions') # TODO remove
        # for column in range(self.h):
        for column, frequency in enumerate(fla_max_fre):
            # TODO ask Xilin why is this condition necessary
            if frequency != 0:
                v1, v1_opinion = self.map_action(column)

                # change innate opinion by max action
                op_max = change_k_innate_opinion(self.s, v1, v1_opinion)
                # op_max = fn_benchmark( # TODO remove benchmark
                #     lambda: self._change_k_innate_opinion(self.s, v1, max_opinion),
                #     label=f"Change innate opinion for {v1} to {max_opinion}",
                #     display=DISPLAY_BENCHMARK,
                # )

                # Derivate optimal Min's opinion for nodeset v2
                # {sum}{j}(s_j(h_j -c))  - rest of terms
                M_rest = self._sum_rest(op_max, v2)
                # M_rest = fn_benchmark( # TODO remove benchmark
                #     lambda: self._sum_rest(op_max, v2),
                #     label=f"_sum_rest: Sum rest of terms for {v2}",
                #     display=DISPLAY_BENCHMARK,
                # )
                weight_M += frequency * M_rest  # {sum}{v} p_v * M

        # Got optimal Min's opinion for v2
        # give a set of k weighted opinions
        derivative_k_opinion = self._k_derivate_s(v2, weight_M)
        # k_opinion = fn_benchmark(
        #     lambda: self._k_derivate_s(v2, weight_M),
        #     label=f"_k_derivate_s: Derivate optimal Min's opinion for {v2}",
        #     display=DISPLAY_BENCHMARK,
        # )

        derivative_k_opinion = [0 if x < 0
                                else 1 if x > 1
                                else x
                                for x in derivative_k_opinion]

        (mixed_por, payoff_row) = self._mixed_k_min_polarization(
            v2, derivative_k_opinion, fla_max_fre)
        # (mixed_por, payoff_row) = fn_benchmark( # TODO remove benchmark
        #     lambda: self._mixed_k_min_polarization(v2, derivative_k_opinion, fla_max_fre),
        #     label=f"_mixed_k_min_polarization: Calculate polarization of minimizer's Mixed Strategy",
        #     display=DISPLAY_BENCHMARK,
        # )

        return (v2, derivative_k_opinion, payoff_row, mixed_por)

    ## PUBLIC METHODS ##

    def setK(self, k: int):
        self.k = k

    def map_action(self, column: int):
        """
        Return the k_node and k_opinion from the column index
        """
        k_opinions, len_kops = max_k_opinion_generator(self.k)
        k_node_index = int(column / len_kops)
        k_node = get_k_node(k_node_index, self.n, self.k)
        k_opinion_index = column % len_kops
        k_opinion = next(islice(k_opinions, k_opinion_index, None))
        return (k_node, k_opinion)

    def run(self, game_rounds: int, memory: int):
        payoff_matrix = np.empty((0, self.h), float)
        max_touched = []
        min_touched = []
        min_touched_all = []
        min_touched_last_100 = []
        max_history = np.zeros(self.h, int)
        # list of tuples of (k_node tuple, k_opinion tuple)
        min_history = []
        max_history_last_100 = np.zeros(self.h, int)
        min_history_last_100 = []
        max_history_column = []

        # Game start from maximizer random play
        (v1, max_opinion, max_pol, column) = self._k_random_play()

        max_history_column.append(column)

        # Save the first action to report later
        first_max = (v1, max_opinion, max_pol)

        # maximizer play history
        max_history[column] += 1

        # maximizer play's frequency, only played 1 time so far, divided by 1
        fla_max_fre = max_history / 1

        # if game start from minimizer random play - make sure two random play are not same agent!!!
        # print('Minimizer first selection')
        (v2, min_opinion, min_pol) = self._k_random_play_1(v1)

        first_min = (v2, min_opinion, min_pol)

        min_touched.extend(v2)
        min_touched_all.append(tuple(v2))

        # store minimizer play history
        min_history.append((tuple(v2), tuple(min_opinion)))
        # print(f'min_history {min_history}')

        # a dictionary of {(v2, min_option): count of this choice}
        min_history_counter = collections.Counter(min_history)

        # frequency of all min options in order
        fla_min_fre = np.array(list(min_history_counter.values()))/1.0

        (_, payoff_row) = self._mixed_k_min_polarization(
            v2, min_opinion, fla_max_fre)
        payoff_matrix = np.vstack([payoff_matrix, payoff_row])

        equi_min = min_pol
        equi_max = max_pol

        for i in range(1, game_rounds + 1):
            print("-" * 20)
            print(f"Game {i}")
            print("-" * 10)

            # print(f'min_history {min_history}')
            # print(f'max_history {max_history}')

            # Reset the history to collect the last 100 rounds
            if i == game_rounds - 99:
                # Remove max frequency less than 0.1--
                max_history_last_100 = np.zeros(self.h)
                min_history_last_100 = []
                min_touched_last_100 = []

            # MAXimizer play
            (v1, max_opinion, equi_max, column) = fn_benchmark(
                # lambda: self._max_k_play(
                lambda: self._max_k_play_multi(
                    payoff_matrix, fla_min_fre, min_touched),
                label="Maximizer play",
                display=DISPLAY_BENCHMARK,
            )
            max_touched = self._push(max_touched, v1, memory)

            # accumulate strategy
            max_history[column] += 1
            max_history_last_100[column] += 1
            max_history_column.append(column)

            # update maximizer frequency after playing
            # +1 to include the initial play before the loop
            fla_max_fre = max_history/(i+1)
            # print(f'fre_max at spot {fla_max_fre[column]}')

            # MINimizer play
            # (v2, payoff_row, min_opinion, equi_min) = self._mixed_min_play(
            #     max_touched, fla_max_fre)
            (v2, min_opinion, payoff_row, equi_min) = fn_benchmark(
                lambda: self._mixed_min_play(max_touched, fla_max_fre),
                label="Minimizer play",
                display=DISPLAY_BENCHMARK,
            )

            min_touched = self._push(min_touched, v2, memory)
            min_touched_all.append(v2)
            min_touched_last_100.append(v2)

            min_history_entry = (v2, tuple(min_opinion))
            # if this is a new min option, append to payoff matrix
            if min_history_entry not in min_history_counter:
                payoff_matrix = np.vstack([payoff_matrix, payoff_row])

            min_history.append(min_history_entry)
            min_history_last_100.append(min_history_entry)

            min_history_counter.update([min_history_entry])

            # frequency of all min options in order
            fla_min_fre = np.array(list(min_history_counter.values()))/(i+1)
            # print(f'fla_min_fre {fla_min_fre.shape}: {fla_min_fre}')

            equi_max = round(equi_max, PRECISION)
            equi_min = round(equi_min, PRECISION)
            if equi_max == equi_min:
                print(
                    f"Reached Nash Equilibrium at round {i} and Equi_Por = {equi_min}")
                # print(f'max_distribution {max_frequency}')
                # print(f'min_distribution {fla_min_fre}')
                break
            else:
                print(
                    f"Not Reached Nash Equilibrium at Equi_Min = {equi_min} and Equi_Max = {equi_max}")

        # Game has finished
        print('MAX_last_100,  all')
        max_l100_fre = max_history_last_100/100
        max_fre = max_history / (game_rounds + 1)
        print(max_l100_fre[np.nonzero(max_l100_fre)],
              max_fre[np.nonzero(max_fre)])
        print(np.nonzero(max_l100_fre)[0],
              np.nonzero(max_fre)[0])

        columns = list(np.nonzero(max_l100_fre)[0])
        for column in list(columns):
            k_opinions, len_kops = max_k_opinion_generator(self.k)
            nodeset_index = int(column/len_kops)
            opset_index = column % len_kops
            k_nodes = get_k_node(nodeset_index, self.n, self.k)
            opinions = next(islice(k_opinions, opset_index, None))
            print(f'Max Nodes: {k_nodes} | Opinion: {opinions}')

        # MINimizer's Strategy in the last 100 round
        min_history_last_100_counter = collections.Counter(
            min_history_last_100)
        fla_min_fre_last_100 = np.array(
            list(min_history_last_100_counter.values()))/100
        print('MIN_last_100,  all')

        # frequency of all min options in order
        fla_min_fre = np.array(
            list(min_history_counter.values()))/(game_rounds + 1)
        print(fla_min_fre_last_100, fla_min_fre)
        print(min_history_last_100_counter, min_history_counter)
        print(f'Max Pol: {equi_max} Min Pol: {equi_min}')

        return GameResult(
            game_rounds=game_rounds,
            first_max=first_max,
            first_min=first_min,
            min_touched=min_touched,
            max_touched=max_touched,
            payoff_matrix=payoff_matrix,
            min_touched_last_100=min_touched_last_100,
            min_touched_all=min_touched_all,
            fla_min_fre=fla_min_fre_last_100,
            fla_max_fre=fla_max_fre,
            min_history=min_history,
            max_history=max_history,
            min_history_last_100=min_history_last_100,
            max_history_last_100=max_history_last_100,
            equi_max=equi_max,
            equi_min=equi_min,
        )


def exportGameResult(network: str, game: Game, result: GameResult, k, memory, experiment):
    # Create the 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    pd.DataFrame(result.payoff_matrix).to_csv(
        f'results/network-{network}-k-{k}-experiment-{experiment}-memory-{memory}-payoff_matrix.csv')

    # Save the original standard output
    original_stdout = sys.stdout

    with open(f'results/network-{network}-k-{k}-experiment-{experiment}-memory-{memory}-results.txt', "w") as f:
        # Change the standard output to the file we created.
        sys.stdout = f

        print('Initial Condition -(agent, opinion, pol)')
        # print(f'Innate op {s}')
        # print(f'Adjacency matrix {G}')
        # print('Selected Nodeset, k_Opinions, Steady-state polarization')
        print(f'Max:\t{result.first_max}')
        print(f'Min:\t{result.first_min}')

        print('_____________________')
        print(f'Max Pol:\t{result.equi_max}')
        print(f'Min Pol:\t{result.equi_min}')

        # MAXimizer's distribution of LAST 100 iteration
        print('Max_distribution_last_100')
        max_l100_fre = result.max_history_last_100/100
        print(max_l100_fre[np.nonzero(max_l100_fre)])
        # print for small network
        # print(max_history_last_100)
        # # Print for Large Network
        print(np.nonzero(max_l100_fre))

        columns = np.nonzero(max_l100_fre)
        columns = list(columns[0])
        for column in columns:
            (k_nodes, opinions) = game.map_action(column)
            print(f'Max Nodes: {k_nodes}\tOpinion: {opinions}')

        print('Max_distribution_all')
        max_fre = result.max_history / (result.game_rounds + 1)
        print(max_fre[np.nonzero(max_fre)])
        print([np.nonzero(max_fre)])

        # MINimizer's Strategy in the last 100 round
        min_touched_last_100_counter = collections.Counter(
            result.min_touched_last_100)
        # frequency of all min options in order
        fla_min_fre = np.array(
            list(min_touched_last_100_counter.values()))/(100)
        print('Min_distribution_last_100')
        print(fla_min_fre)
        print(min_touched_last_100_counter)
        # print(min_touched_last_100)

        # a dictionary include {'min_option': count of this choice}
        min_touched_all_counter = collections.Counter(result.min_touched_all)
        # frequency of all min options in order
        fla_min_fre_1 = np.array(
            list(min_touched_all_counter.values())) / (result.game_rounds + 1)
        print('Min_distribution_all')
        print(fla_min_fre_1)
        print(min_touched_all_counter)
        np.set_printoptions(precision=3)

        # a dictionary include {'min_option': count of this choice}
        min_history_counter = collections.Counter(result.min_history)
        print(f'min_history_counter:\t{min_history_counter}')

        print(f'min_recent_{memory}_touched')
        print(result.min_touched)
        print(f'max_recent_{memory}_touched')
        print(result.max_touched)

        # Reset the standard output to its original value
        sys.stdout = original_stdout


def len_actions(k: int, n: int) -> int:
    """
    Calculate the length of all possible actions for k opinions and n nodes
    All possible actions = all combination of k nodes * all permutations of k opinions (ordered)
    """
    possible_max_opinions = [0, 1]
    # number of opinion permutations
    len_kops = len(possible_max_opinions) ** k
    # Horizontal length of all possible actions
    h = comb(n, k) * len_kops
    return h


def max_k_opinion_generator(k, calculate_size=True):
    '''
    Return a generator for all permutations of k opinions
    '''
    max_option = [0, 1]
    size: int = len(max_option) ** k if calculate_size else None
    return product(max_option, repeat=k), size


# [x] improved
def change_k_innate_opinion(op, k_node: list | tuple, k_opinion: list | tuple):
    '''
    Return a new copy of op that has the opinions of k_nodes changed to k_opinion
    '''
    new_op = np.copy(op)
    for (index, node) in enumerate(k_node):
        new_op[node] = k_opinion[index]
    return new_op


def convert_available(k, k_nodes: list, touched: list):
    """
    - k: number of nodes in k_nodes
    - k_nodes: list of k nodes (sorted)
    - touched: list of touched nodes

    Return the next available k nodes
    """
    touched.sort()
    for i in range(k):
        for t in touched:
            max = t
            for j in range(i, k):
                if k_nodes[j] == max:
                    k_nodes[j] += 1
                    max = k_nodes[j]
                else:
                    break
    return k_nodes


def find_index(n: int, k_node: list):
    """
    - n: number of nodes

    Return the k_nodes's index in list of all combinations
    """
    latter = 0
    index = 0
    k = len(k_node)

    for i in range(k):
        before = k_node[i] + 1
        L_min = latter + 1
        L_max = before - 1

        M = L_max - L_min

        for m in range(1, M+2):
            P = n - latter - m
            L = k - 1 - i
            index = index + comb(P, L)
        latter = before

    return index


def get_k_node(i: int, n: int, k: int):
    """
    Returns the i-th k_node among C(n, k)
    """
    k_node = []
    r = i+0
    j = -1
    for s in range(1, k+1):
        cs = j+1
        while r-comb(n-1-cs, k-s) >= 0:
            r -= comb(n-1-cs, k-s)
            cs += 1
        k_node.append(cs)
        j = cs
    return k_node
