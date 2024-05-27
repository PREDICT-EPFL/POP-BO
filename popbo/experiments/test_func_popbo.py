import sys
sys.path.append('..')

import datetime
import util
from popbo import POPBO
import sigma
import numpy as np
import sys
import time
import os
import pickle

test_prob_name_list = ['Branin', 'CurrinExp', 'Beale',
        'Styblinski_tang', 'Bukin', 'Cross_in_tray', 'Eggholder',
        'Holder_table', 'Langerman', 'Levy13', 'Shubert']
test_prob_names_to_run = [test_prob_name_list[10]]
# test_prob_name_list[3:] #'Branin' #'CurrinExp' #'Forrester'


def run_one_instance_popbo(problem_dim, num_run, config):
    popbo = POPBO(config)
    f = config['obj']

    x_prime = [np.random.rand()]
    x = popbo.query_new_point(x_prime=x_prime)

    eval_data_list = []
    for k in range(num_run):
        x_prime = x
        x = popbo.query_new_point(x_prime=x_prime)
        one = sigma.pref_oracle(f(x), f(x_prime))
        popbo.add_new_data(x_prime=x_prime, x=x, one=one)
        eval_data_list.append([x, f(x)])

    data_list = []
    for k in range(num_run):
        data_list.append(
            [
                [
                    f(popbo.x_list[k]),
                    f(popbo.x_prime_list[k]),
                    popbo.one_list[k]]
            ]
        )

    opt_val = config['opt_val']
    return eval_data_list, data_list, opt_val

def run_one_instance_popbo_best_x(problem_dim, num_run, config):
    popbo = POPBO(config)
    f = config['obj']

    bounds = config['bounds']

    if problem_dim == 1:
        x_prime = [
            np.random.rand() * (bounds[1] - bounds[0]) + bounds[0]
        ]
    elif problem_dim == 2:
        x_prime = [
            np.random.rand() * (bounds[0][1] - bounds[0][0]) + bounds[0][0],
            np.random.rand() * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
        ]

    x = popbo.query_new_point(x_prime=x_prime)
    opt_val = config['opt_val']
    cumu_reg = (opt_val - f(x_prime)) + (opt_val - f(x))
    eval_data_list = []
    for k in range(num_run):
        x_prime = x

        x = popbo.query_new_point(x_prime=x_prime)
        proj_x = [0] * problem_dim
        if problem_dim == 1:
            proj_x = max(min(x, bounds[1]), bounds[0])
        else:
            for j in range(problem_dim):
                proj_x[j] = max(min(x[j], bounds[j][1]), bounds[j][0])

        x = proj_x
        one = sigma.pref_oracle(f(x), f(x_prime))
        popbo.add_new_data(x_prime=x_prime, x=x, one=one)
        eval_data_list.append([x, f(x)])
        cumu_reg += (opt_val-f(x))
        print(f'Step {k}, instantaneous regret: {opt_val-f(x)},'+
              f'cumu_reg: {cumu_reg}.\n')


    data_list = []
    for k in range(num_run):
        data_list.append(
            [
                [f(popbo.x_list[k]),
                 f(popbo.x_prime_list[k]),
                 popbo.one_list[k]]
            ]
        )

    opt_val = config['opt_val']

    best_x_list = popbo.estimated_best_x_list
    best_x_val_list = popbo.estimated_best_x_val_list
    return eval_data_list, data_list, opt_val, best_x_list, best_x_val_list

for test_prob_name in test_prob_names_to_run:
    eval_data_list_list_popbo = []
    data_list_list_popbo = []
    best_x_list_list_popbo = []
    best_x_val_list_list_popbo = []
    last_best_x_regret_list_list_popbo = []

    opt_val_list = []

    eval_data_list_list_qEUBO = []
    data_list_list_qEUBO = []

    eval_data_list_list_sgp = []
    data_list_list_sgp = []

    eval_data_list_list_GJ = []
    data_list_list_GJ = []

    instance_num = 30
    num_run = 30
    problem_dim = 2

    for k in range(instance_num):
        print(f'{test_prob_name}: {k}-th instance.')
        config = util.get_config('test_func_'+test_prob_name,
                                problem_dim=problem_dim)

        eval_data_list_popbo, data_list_popbo, opt_val, best_x_list_popbo, \
            best_x_val_list_popbo = run_one_instance_popbo_best_x(
            problem_dim, num_run, config
        )

        eval_data_list_list_popbo.append(eval_data_list_popbo)
        data_list_list_popbo.append(data_list_popbo)

        best_x_list_list_popbo.append(best_x_list_popbo)
        best_x_val_list_list_popbo.append(best_x_val_list_popbo)
        last_best_x_regret_list_list_popbo.append(
            opt_val - best_x_val_list_popbo[-1]
        )
        opt_val_list.append(opt_val)
        last_best_regret_mean = np.mean(last_best_x_regret_list_list_popbo)
        last_best_regret_std = np.std(last_best_x_regret_list_list_popbo)
        print(
            f'{k} instances: {last_best_regret_mean}, {last_best_regret_std}'
        )

    now_time_str = datetime.datetime.now().strftime(
            "%H_%M_%S-%b_%d_%Y")


    with open(f'../result/test_func_popbo_{test_prob_name}_' +
            f'{now_time_str}.pkl', 'wb') as f:
        pickle.dump([
            eval_data_list_list_popbo, data_list_list_popbo, opt_val_list,
            best_x_list_list_popbo, best_x_val_list_list_popbo
            ], f)
