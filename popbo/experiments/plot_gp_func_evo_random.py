import sys
sys.path.append('..')
import datetime
import util
from popbo import POPBO
import sigma
import numpy as np
import pickle
import random


BETA_0 = 0.2
Is_Const_Beta = False
if Is_Const_Beta:
    BETA = lambda t: BETA_0  # 1.0
else:
    BETA = lambda t: BETA_0 * np.sqrt(t+1)


def get_random_plot_data(batch_num=3, iter_per_batch=5):
    config = util.get_config('GP_sample_func', problem_dim=1)
    popbo = POPBO(config)
    f = config['obj']
    x_prime = [np.random.rand()]
    x = popbo.query_new_point(x_prime=x_prime)

    x_grid = popbo.config['x_grid']
    outer_iter_id_to_data = dict()
    for batch_id in range(batch_num):
        for int_iter_id in range(iter_per_batch):
            x_prime = x
            x = random.sample(x_grid.tolist(), 1)[0]
            one = sigma.pref_oracle(f(x), f(x_prime))
            popbo.add_new_data(x_prime=x_prime, x=x, one=one)

        max_ll_list = []
        upb_list = []
        lpb_list = []
        f_list = []
        sample_x_list = popbo.x_list
        out_iter_id = batch_id * iter_per_batch + int_iter_id
        for x in x_grid:
            max_ll_list.append(popbo.get_maxLL_min_norm_inter(x))
            upb_list.append(popbo.get_upper_bound(x, beta=BETA(out_iter_id)))
            lpb_list.append(popbo.get_lower_bound(x, beta=BETA(out_iter_id)))
            f_list.append(f(x))

        outer_iter_id_to_data[out_iter_id] = [
            x_grid,
            sample_x_list,
            f_list,
            max_ll_list,
            upb_list,
            lpb_list
        ]

    return outer_iter_id_to_data


batch_num = 3
iter_per_batch = 10
instance_total_num = 5

data_dict_list = []

for instance_id in range(instance_total_num):
    data_dict = get_random_plot_data(
        batch_num=batch_num, iter_per_batch=iter_per_batch
    )
    data_dict_list.append(data_dict)

now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")

with open('../result/GP_random_plot_data_' +
          f'{now_time_str}_BETA_{BETA_0}_{Is_Const_Beta}.pkl', 'wb') as f:
    pickle.dump(data_dict_list, f)
