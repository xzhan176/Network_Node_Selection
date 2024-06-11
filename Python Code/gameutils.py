import numpy as np
import copy
import random
from utils import *

def push(objs, element, memory):
    if len(objs) >= memory:
        objs.pop(0)
        print('pop')
    objs.append(element)
    return objs

def row_index(v2, min_opinion):
    row = 11*v2 + min_opinion*10
    return int(row)

def column_index(v1, max_opinion):
    column = 2*v1 + max_opinion
    return int(column)  # the python data frame index

def random_play(s, n, A, L):
    """
    Player randomly chooses an agent and randomly change the agent
    """
    ops = copy.copy(s)
    # randomly select an agent index
    v_index = random.randint(0, n-1)
    # randomly select an opinion between 0 and 1
    # TODO confirm which one is correct
    # new_opinion = random.randint(0, 1)
    new_opinion = random.uniform(0, 1)
    # print(f'new_op: {new_opinion}')

    # Store old opinion
    old_opinion = ops[v_index, 0]

    # update the opinion
    ops[v_index, 0] = new_opinion
    # print('Only 1 opinion changed')
    # print(op)
    print(f"    Agent{v_index}'s opinion {old_opinion} changed to {new_opinion}")
    polar = obj_polarization(A, L, ops, n)

    # restore op array to innate opinion
    ops[v_index] = old_opinion
    print(f"Network reaches equilibrium Polarization: {polar}")
    # print('Should be restored')
    # print(op)
    return (v_index, new_opinion, polar)

def make_payoff_row(op1, v2, s, n, A, L):
    payoff_row = np.zeros(2*n)

    # print('one opinion changed -min')
    # print(op1)
    for column in range(2*n):
        # print(column)
        v1 = int(column/2)  # i.e., column 11 is agent 5, opinion 1
        max_opinion = column % 2
        # print(v1, max_opinion)
        # update the maximizer's change to the opinion array that has changed by minimizer(op1)
        op2 = copy.copy(op1)
        # temp = op1[v1]
        op2[v1, 0] = max_opinion

        # calculate the polarization with both max and min's action
        payoff_row[column] = obj_polarization(A, L, op2, n)
        # op1[v1,0] = temp # restore
        # print(op2,payoff_row[column])

    # when v1 == v2, the polarization should be negative for max, infinite for min.
    ####################### ZERO SUM when v1==v2, the polarization is innate polarization########################
    j_1 = 2*v2 + 0
    j_2 = 2*v2 + 1
    O_P = obj_polarization(A, L, s, n)
    payoff_row[j_1] = O_P
    payoff_row[j_2] = O_P

    return payoff_row

def make_payoff_row(op1, v2, s, n, A, L):
    payoff_row = np.zeros(2*n)

    # print('one opinion changed -min')
    # print(op1)
    for column in range(2*n):
        # print(column)
        v1 = int(column/2)  # i.e., column 11 is agent 5, opinion 1
        max_opinion = column % 2
        # print(v1, max_opinion)
        # update the maximizer's change to the opinion array that has changed by minimizer(op1)
        op2 = copy.copy(op1)
        # temp = op1[v1]
        op2[v1, 0] = max_opinion

        # calculate the polarization with both max and min's action
        payoff_row[column] = obj_polarization(A, L, op2, n)
        # op1[v1,0] = temp # restore
        # print(op2,payoff_row[column])

    # when v1 == v2, the polarization should be negative for max, infinite for min.
    ####################### ZERO SUM when v1==v2, the polarization is innate polarization########################
    j_1 = 2*v2 + 0
    j_2 = 2*v2 + 1
    O_P = obj_polarization(A, L, s, n)
    payoff_row[j_1] = O_P
    payoff_row[j_2] = O_P

    return payoff_row

def mixed_min_polarization(s, v2, weight_op, fla_max_fre):
    """
    Calculate polarization of minimizer's Mixed Strategy
    """

    op1 = copy.copy(s)  # make a copy of the innate opinion array
    op1[v2, 0] = weight_op  # then only updated by minimizer's current change
    # print('Min update')
    # print(v2, weight_op)

    # calculate the polarization with both min(did here) and max's action(in make_payoff_row)
    # the vector list out 2*n payoffs after min's action combine with 2*n possible max's actions
    payoff_row = make_payoff_row(op1, v2)
    # print(payoff_row)

    # Replace the the column_index of agent v2 with 100 for min - NO NEED TO REPLACE NOW

    # print('Min Payoff Row')
    # print(payoff_row)

    # calculate fictitious payoff - equi_min
    # fla_max_fre recorded the frequency of each maximizer's action, frequency sum = 1
    payoff_cal = payoff_row * fla_max_fre
    # payoff (2*n array) * maximizer_action_frequency (2*n array)

    # add up all, calculate average/expected payoff
    mixed_pol = np.sum(payoff_cal)

    # print('min_mixed_polarization')
    # print(mixed_pol)
    # Replace the the column_index of agent v2 with 100 for min

    return (mixed_pol, payoff_row)

def derivate_s(op, n, A, v2):
    # op - opinion array that updated by maximizer
    c = [1/n] * n
    # print(c)
    sum_term = 0
    j = 0

    sum_term = np.dot(np.dot((A-c), (A[v2]-c)), op)  # sum up all terms

    # exclude the term that j = v2
    term_out = op[v2] * np.dot((A[v2]-c), (A[v2]-c))
    sum_s = sum_term - term_out    # numerator

    s_star = -sum_s/np.dot((A[v2]-c), (A[v2]-c))
    s_star = s_star[0]  # take value out of array
    min_opinion = min(max(0, s_star), 1)

    # print('Min opinion-should be unique')
    # print(min_opinion)
    return min_opinion

def min_mixed_opinion_1(s, n, A, L, v2, fla_max_fre):
    weight_op = 0

    # loop for each max_action(in total 2*n)
    for column in range(2*n):

        if fla_max_fre[column] != 0:
            v1 = int(column/2)  # i.e., column 11 is agent 5, opinion 1
            max_opinion = column % 2
            op = copy.copy(s)
            op[v1] = max_opinion
            # print(op)
            # print('Weight')
            # print(fla_max_fre[column])

            # find min_s_star for each max_action
            min_opinion = derivate_s(op, n, A, v2)
            op1 = copy.copy(op)
            # after max action, update min action on opinion array
            op1[v2] = min_opinion
            # print(min_opinion)
            min_por = obj_polarization(A, L, op1, n)
            t = 0
            weight_op += fla_max_fre[column]*min_opinion  # sum up p_i*s_i

    (mixed_por, payoff_row) = mixed_min_polarization(s, n, A, L, v2, weight_op, fla_max_fre)
    # print('Weighted polarization')
    # print(mixed_por)

    return (weight_op, payoff_row, mixed_por)

def mixed_min_polarization(s, n, A, L, v2, weight_op, fla_max_fre):
    """
    Calculate polarization of minimizer's Mixed Strategy
    """

    op1 = copy.copy(s)  # make a copy of the innate opinion array
    op1[v2, 0] = weight_op  # then only updated by minimizer's current change
    # print('Min update')
    # print(v2, weight_op)

    # calculate the polarization with both min(did here) and max's action(in make_payoff_row)
    # the vector list out 2*n payoffs after min's action combine with 2*n possible max's actions
    payoff_row = make_payoff_row(op1, v2, s, n, A, L)
    # print(payoff_row)

    # Replace the the column_index of agent v2 with 100 for min - NO NEED TO REPLACE NOW

    # print('Min Payoff Row')
    # print(payoff_row)

    # calculate fictitious payoff - equi_min
    # fla_max_fre recorded the frequency of each maximizer's action, frequency sum = 1
    payoff_cal = payoff_row * fla_max_fre
    # payoff (2*n array) * maximizer_action_frequency (2*n array)

    # add up all, calculate average/expected payoff
    mixed_pol = np.sum(payoff_cal)

    # print('min_mixed_polarization')
    # print(mixed_pol)
    # Replace the the column_index of agent v2 with 100 for min

    return (mixed_pol, payoff_row)

def mixed_choose_min_vertex(s, n, A, L, v1, max_opinion, fla_max_fre):
    """Minimizer search: Go through each agent"""
    # current polarization that changed by maximizer, "innate" objective that min start with
    op = copy.copy(s)
    op[v1, 0] = max_opinion
    # print('Check if op has been updated by Maximizer')
    # print(op)
    # min_por- set a standard to compare with pol after min's action
    maxup_por = obj_polarization(A, L, op, n)
    min_por = 1000  # use the infinite big min_por
    # print('check maxup por')
    # print(maxup_por)
    # payoffs = []    # create an empty list to store all polarizations
    champion = (None, None, 0, None)  # assume the best action is champion

    for v2 in range(n):
        # print('_________________________________')
        # print('Min start with agent '+ str(v2) )
        ################################# for ZERO SUM ##########################################
        if v2 == v1:
            (por, payoff_row) = mixed_min_polarization(s, n, A, L, v2, s[v2], fla_max_fre)
            # doesn't change the innate opinion, keep the polarization as innate polarization
            changed_opinion = s[v2, 0]
            # print('Weighted polarization')
            # print(por)
            # print("Maxup por, min_por:", maxup_por, por)
        # #########################################################################################
        else:
            # find the best new_op option
            (changed_opinion, payoff_row, por) = min_mixed_opinion_1(s, n, A, L, v2, fla_max_fre)
            # print(changed_opinion, por, maxup_por)

            # print("Maxup por, min_por:", maxup_por, por)
        if por < min_por:  # if the recent polarization is smaller than the minimum polarization in the history
            min_por = por
            # update the recent option as champion
            champion = (v2, changed_opinion, payoff_row, min_por)
            # print("Maxup por, min_por:", min_por, por)
        # else:
            # print('Innate polarization is smaller than Min action')

    # find the best minimizer's action after going through every new_op option of every agent
    return champion

def mixed_min_play(s, v1, max_opinion, n, A, L, fla_max_fre):
    """
    Op has been updated by maximizer, fla_max_fre includes max's history,
    so minimizer react to the innate op after that
    """

    # print('_______________________')
    # print('Minimizer Play')
    # print('Only 1 opinion changed')
    # print(op)

    min_champion = mixed_choose_min_vertex(s, n, A, L, v1, max_opinion, fla_max_fre)
    (v2, min_opinion, payoff_row, min_pol) = min_champion

    if v2 == None:    # if minimizer cannot find a action to minimize polarization after maximizer's action
        print('Minimizer fail')

    else:
        print("                                ")
        print("Minimizer finds its target agent:")
        # print('v2', 'changed_opinion', 'innate_obj', 'obj')
        # print(v2, min_opinion, innat_equi_por, min_pol)

        # Store innate_op of the min_selected vertex
        # old_opinion_min = op[v2, 0]
        old_opinion_min = s[v2, 0]

        print(f"    Agent{v2}'s opinion {old_opinion_min} changed to {min_opinion}")
        # print('fla_max_fre')
        # print(np.nonzero(fla_max_fre))
        # print(fla_max_fre [np.nonzero(fla_max_fre)])

        # print("Payoff row")
        # print(payoff_row)
        # print(f"Network reaches equilibrium Polarization: {str(min_pol)}")
        # print('2 opinion changed')
    return (v2, payoff_row, min_opinion, min_pol)

def mixed_max_polarization(payoff_matrix, v1, max_opinion, fla_min_fre):
    """
    Op has been updated by minimizer, fla_min_fre includes min's history, so maximizer react to the innate op after that
    """

    # create payoff matrix for maximizer
    column = int(column_index(v1, max_opinion))
    # print(payoff_matrix)
    # print(f"column {str(column)}")
    payoff_vector = payoff_matrix[:, column]

    # print(f'payoff vector: {payoff_vector}')

    # calculate fictitious payoff - equi_max
    payoff_cal = payoff_vector * fla_min_fre  # payoff * frequency

    # print(f'max_payoff_calculation: {payoff_cal}')
    mixed_pol = np.sum(payoff_cal)  # add up
    # print(f"Max_mixed_polarization: {mixed_pol}")

    return mixed_pol

def max_mixed_opinion(payoff_matrix, n, v1, fla_min_fre):
    """
    determines if value of opinion at v should be set to 0 or 1 to maximize equilibrium polarization
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
        por_arr[j] = mixed_max_polarization(payoff_matrix, v1, max_opinion, fla_min_fre)
        j = j + 1   # index increase 1, put the polarization in array

    # print(f'Polarization Options {por_arr}')

    # the index of maximum polarization = max_opinion --[0,1]
    maximize_op = np.argmax(por_arr)
    # find the maximum polarization in the record
    max_por = np.max(por_arr)

    # print('new_op', 'innat_equi_por', 'max_por')
    # print(maximize_op, innat_equi_por, max_por)

    return (maximize_op, max_por)

def mixed_choose_max_vertex(payoff_matrix, op, s, n, A, L, v2, fla_min_fre):
    """
    determine which agent maximizer should select to maximizer the equilibrium polarization
    """
    # print('Check if op has been updated by minimizer')
    # print(op)
    # use "innate"(after min action) polarization as a comparable standard to find max_por
    max_por = 0

    # store innate min_update polarization
    minup_por = obj_polarization(A, L, op, n)
    # print('check minup por')
    # print(minup_por)

    champion = (None, None, max_por)  # assume champion is the best action
    for v1 in range(n):
        # print('Maximizer start from agent'+str(v1))
        (changed_opinion, por) = max_mixed_opinion(payoff_matrix, n, v1, fla_min_fre)

        if v2 == v1:
            # doesn't change the innate opinion, keep the polarization as innate polarization
            changed_opinion = s[v2, 0]

            # print('changed_opinion, por, minup_por')

        if por > max_por:  # if the polarization of most recent action > maximum polarization of previous actions
            max_por = por
            # save the this action as champion
            champion = (v1, changed_opinion, max_por)
            # else:
                # print('Innate polarization is bigger than max action')

    return (champion)

def mixed_max_play(payoff_matrix, s, n, A, L, v2, min_opinion, fla_min_fre):
    op = copy.copy(s)   # op is a copy of innate opinion

    # update innate opinion
    # Op has been updated by minimizer, so maximizer react to the innate op after that
    op[v2, 0] = min_opinion

    # The best choice among all opinions and vertices
    max_champion = mixed_choose_max_vertex(payoff_matrix, op, s, n, A, L, v2, fla_min_fre)
    (v1, max_opinion, max_pol) = max_champion

    if v1 == None:
        print('Maximizer fail')

    else:
        print("                                ")
        print("Maximizer finds its target agent:")
        # Store innate_op of the max_selected vertex
        old_opinion_max = op[v1, 0]

    if v1 == v2:
        # If select the same agent, doesn't change the opinion
        max_opinion = s[v1, 0]

        # check if agent's opinion is changed or not
    print(f"    Agent{v1}'s opinion {old_opinion_max} changed to {max_opinion}")
    # print(f"Network reaches equilibrium Polarization: {str(max_pol)}")
    # print('2 opinion changed')
    # print(op)

    return (v1, max_opinion, max_pol)