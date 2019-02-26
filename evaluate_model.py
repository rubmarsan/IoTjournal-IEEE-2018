import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pickle
from gym_lora_faster import LoRaWorld
import random
import sys
# import tensorflow as tf
import os
import warnings
import platform
from sklearn.neural_network import MLPClassifier
from scipy import stats
import glob
from time import time
from es import PEPG

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

global lambdas_, lengths, priorities, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, starting_point, FORCE_GLOBAL_ONE, suboptimal
global fig1, fig2, fig3, ax1, ax2, ax3

if platform.node() == "alioth":
    print('Alioth config')
    WORKERS = 38
    REPS = 1
    TEST_BASELINE = False
else:
    print('Laptop config')
    WORKERS = 8
    REPS = 1
    TEST_BASELINE = False


def predict(x):
    return np.argmax(np.random.multinomial(1, x))


def sample_action_simple(X, weights):
    v = np.tanh(weights.T.dot(X))
    return predict([v, 1 - v])


def sample_action(X, weights):
    weights_1 = weights[:weights_1_n]
    weights_1.shape = (D, h_1_size)
    bias_1 = weights[weights_1_n:weights_1_n + bias_1_n]
    weights_2 = weights[weights_1_n + bias_1_n:weights_1_n + bias_1_n + weights_2_n]
    weights_2.shape = (h_1_size, h_2_size)
    bias_2 = weights[weights_1_n + bias_1_n + weights_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n]
    weights_3 = weights[
                weights_1_n + bias_1_n + weights_2_n + bias_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n]
    weights_3.shape = (h_2_size, K)
    bias_3 = weights[weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n:]

    # assert weights.shape[0] == (weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n + bias_3_n)
    # assert X.shape[0] == D

    z_1 = np.tanh(weights_1.T.dot(X) + bias_1)
    z_2 = np.tanh(weights_2.T.dot(z_1) + bias_2)
    Y = 1 / (1 + np.exp(-(weights_3.T.dot(z_2) + bias_3)))
    action = predict([Y[0], 1 - Y[0]])
    return action


def rollout(env, model, std_perturbance=0, force_one=False, suboptimal=False):
    global fig1, fig2, fig3, ax1, ax2, ax3

    if suboptimal:
        observation = env.reset(gen_random=True)
    else:
        observation = env.reset(gen_random=False)

    max_time = 3600 * 12  # 24 * 2

    total_reward = 0
    info_time = 0
    done = False
    rewards = list()
    throughputs = list()
    throughputs.append((0, env.R))
    updated = list()
    when_updated = None

    while (info_time < max_time) and not done:
        first_node_updatable = observation[41] == 1  # 1 => remaining TDC, 40 => updated state of first 40 nodes

        if force_one:
            action = 1
        else:
            observation = env.fixed_length_input(observation, extended=True)  # type: np.ndarray
            observation += np.random.normal(loc=0, scale=std_perturbance, size=(observation.shape[0]))
            observation = np.concatenate([[1], observation])
            action = sample_action(observation, model)

        # action = 1
        if first_node_updatable and action == 1 and when_updated is None and env.last_TDC == env.TDC_MAX:
            when_updated = info_time

        # assert (when_updated is None) !=

        observation, reward, done, info_time = env.step(action)
        updated.append((info_time, np.count_nonzero(observation[1:env.N + 1])))
        rewards.append((info_time, reward))
        throughputs.append((info_time, env.R))

        total_reward += reward

    extra_time = 1
    final_reward = env.gather_last_rewards(til=max_time + extra_time)
    total_reward += final_reward

    rewards.append((max_time + extra_time, final_reward))
    updated.append((max_time + extra_time, updated[-1][1]))
    throughputs.append((max_time + extra_time, throughputs[-1][1]))
    rewards = np.array(rewards)
    updated = np.array(updated)
    throughputs = np.array(throughputs)

    # if len(np.where(np.diff(throughputs[:, 1]) < -1e-1)[0]) == 0:
    #     if FORCE_GLOBAL_ONE:
    #         print('AT anytime')
    #     else:
    #         print('ANN anytime')
    # else:
    #     print('HALT')

    if NUM_TESTS == 1:
        if FORCE_GLOBAL_ONE:
            ax1.plot(rewards[:, 0], np.cumsum(rewards[:, 1]) / 1e3, 'k-', linewidth=2)
        else:
            ax1.plot(rewards[:, 0], np.cumsum(rewards[:, 1]) / 1e3, 'k--', linewidth=2)

        ax1.set_ylabel(r'$P_{\pi}$' + ' in Kbytes', fontsize=16)
        ax1.set_title('Accumulated average throughput per node vs time', fontsize=15)
        ax1.grid(True)
        ax1.set_xlim([0, 35e3])
        ax1.set_ylim([0, 30])
        ax1.tick_params(axis='x', which='major', labelsize=14)
        ax1.tick_params(axis='y', which='major', labelsize=13)
        ax1.set_xticks([10e3, 20e3, 30e3])
        ax1.set_xticklabels(['1 hour', '2 hours', '3 hours'])
        fig1.tight_layout()

        legend_elements = [
            Line2D([0], [0], color='k', lw=2, linestyle='--', label='Proposed policy with ' + r'$C_{opt}$'),
            Line2D([0], [0], color='k', lw=2, linestyle='-', label='Always Update policy with ' + r'$C_{opt}$')
        ]
        ax1.legend(handles=legend_elements, loc=2, fontsize=15)

        if FORCE_GLOBAL_ONE:
            ax2.plot(throughputs[:, 0], throughputs[:, 1], 'k-', linewidth=2)
        else:
            ax2.plot(throughputs[:, 0], throughputs[:, 1], 'k--', linewidth=2)

        ax2.set_ylabel(r'$\Gamma$' ' in bytes/s', fontsize=16)
        ax2.set_title('Average throughput per node vs time', fontsize=15)
        ax2.grid(True)
        ax2.set_xlim([0, 35e3])
        ax2.tick_params(axis='x', which='major', labelsize=14)
        ax2.tick_params(axis='y', which='major', labelsize=13)
        ax2.set_xticks([10e3, 20e3, 30e3])
        ax2.set_xticklabels(['1 hour', '2 hours', '3 hours'])
        ax2.legend(handles=legend_elements, loc=4, fontsize=15)
        fig2.tight_layout()

        if FORCE_GLOBAL_ONE:
            ax3.plot(updated[:, 0], updated[:, 1] / env.N * 100, 'k-', linewidth=2)
        else:
            ax3.plot(updated[:, 0], updated[:, 1] / env.N * 100, 'k--', linewidth=2)

        ax3.set_ylabel('Updated nodes (%)', fontsize=16)
        ax3.set_title('Percentage of updated nodes vs time', fontsize=15)
        ax3.grid(True)
        ax3.set_xlim([0, 35e3])
        ax3.set_ylim([0, 55])
        ax3.tick_params(axis='x', which='major', labelsize=14)
        ax3.tick_params(axis='y', which='major', labelsize=13)
        ax3.set_xticks([10e3, 20e3, 30e3])
        ax3.set_xticklabels(['1 hour', '2 hours', '3 hours'])
        ax3.legend(handles=legend_elements, loc=4, fontsize=15)
        fig3.tight_layout()

    return total_reward, done



def rollout_when_updated(env, model, std_perturbance=0, force_one=False, suboptimal=False):
    global fig1, fig2, fig3, ax1, ax2, ax3

    if suboptimal:
        observation = env.reset(gen_random=True)
    else:
        observation = env.reset(gen_random=False)

    max_time = 3600 * 12  # 24 * 2

    total_reward = 0
    info_time = 0
    done = False
    rewards = list()
    throughputs = list()
    throughputs.append((0, env.R))
    updated = list()
    when_updated = None

    while (info_time < max_time) and not done:
        first_node_updatable = observation[41] == 1  # 1 => remaining TDC, 40 => updated state of first 40 nodes

        if force_one:
            action = 1
        else:
            observation = env.fixed_length_input(observation, extended=True)  # type: np.ndarray
            observation += np.random.normal(loc=0, scale=std_perturbance, size=(observation.shape[0]))
            observation = np.concatenate([[1], observation])
            action = sample_action(observation, model)

        # action = 1
        if first_node_updatable and action == 1 and when_updated is None and env.last_TDC == env.TDC_MAX:
            when_updated = info_time

        # assert (when_updated is None) !=

        observation, reward, done, info_time = env.step(action)
        updated.append((info_time, np.count_nonzero(observation[1:env.N + 1])))
        rewards.append((info_time, reward))
        throughputs.append((info_time, env.R))

        total_reward += reward

    extra_time = 1
    final_reward = env.gather_last_rewards(til=max_time + extra_time)
    total_reward += final_reward

    rewards.append((max_time + extra_time, final_reward))

    return total_reward, done, when_updated


def rollout_rep(params):
    global lambdas_, lengths, priorities, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, REPS, FORCE_GLOBAL_ONE, alpha, suboptimal

    model, seed = params
    random.seed(seed)
    np.random.seed(seed)
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)

    rs = list()
    for rep in range(REPS):
        r, done = rollout(env, model, force_one=FORCE_GLOBAL_ONE, suboptimal=suboptimal)
        rs.append(r)
    rs = np.array(rs)

    # if not FORCE_GLOBAL_ONE:
    #     print('End rollout_rep')

    return rs


def rollout_rep_when_updated(params):
    global lambdas_, lengths, priorities, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, REPS, FORCE_GLOBAL_ONE, alpha, suboptimal

    model, seed = params
    random.seed(seed)
    np.random.seed(seed)
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)

    rs = list()
    whens = list()
    for rep in range(REPS):
        r, done, when_updated = rollout_when_updated(env, model, force_one=FORCE_GLOBAL_ONE, suboptimal=suboptimal)
        rs.append(r)
        whens.append(when_updated)

    rs = np.array(rs)
    whens = np.array(whens)

    return rs, whens


if __name__ == '__main__':
    global lambdas_, lengths, priorities, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, starting_point, FORCE_GLOBAL_ONE, suboptimal
    global fig1, fig2, fig3, ax1, ax2, ax3

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seed = int(sys.argv[1])
        print('Received seed', seed)
        verbose = False
    else:
        seed = 0
        print('Forcing seed', seed)
        verbose = True

    random.seed(seed)
    np.random.seed(seed)

    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        num_nodos = int(sys.argv[2])
        print('Num nodos', num_nodos)
    else:
        num_nodos = 30  # en lugar de 60 nodos con DC = 1% -> 6 con DC = 0.1%?
        print('Num nodos', num_nodos)

    if len(sys.argv) > 3 and sys.argv[3].isdigit() and int(sys.argv[3]) in [0, 1]:
        sub_optimal_C = bool(int(sys.argv[3]))
        print('Sub optimal', sub_optimal_C)
    else:
        sub_optimal_C = False
        print('Sub optimal', sub_optimal_C)

    min_rate = 0.5
    max_rate = 100  # vamos a modelar el T medio entre paquetes, y la lambda como inversa de esto
    lambdas_ = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, num_nodos) + min_rate)
    lengths = np.random.randint(15, 30, (lambdas_.shape[0]))

    min_priority = 0
    max_priority = 1  # 10000
    priorities = (max_priority - min_priority) * np.random.random(num_nodos) + min_priority

    varying_priority = False
    if len(sys.argv) > 4 and float(sys.argv[4]) >= 0:
        warnings.warn("Forcing first node to be of priority: " + sys.argv[4])
        priorities[0] = float(sys.argv[4])
        varying_priority = True


    min_current_priority = 0  # 1
    max_current_priority = 1  # 10000
    priorities_power = (max_current_priority - min_current_priority) * np.random.random(
        num_nodos) + min_current_priority

    SNRs = np.random.randint(-23, 0, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    C_opt = np.zeros((49, lambdas_.shape[0]))
    alpha = 1
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)

    # env.find_optimal_by_adr()
    # env.C = env.C_opt
    # print(env.compute_network_performance())
    #
    # env.find_optimal_c_13()
    # env.C = env.C_opt
    # print(env.compute_network_performance())
    # env.get_reward_matricial_alt(env.C)


    if sub_optimal_C:
        suboptimal = True
        env.find_optimal_by_adr()
        print('Suboptimal configuration under use')
    else:
        suboptimal = False
        if varying_priority:
            path_c_opt = 'c_opt_{}_var.p'.format(priorities[0])
        else:
            path_c_opt = 'c_opt_{}.p'.format(num_nodos)

        if os.path.exists(path_c_opt):
            C_opt_t = pickle.load(open(path_c_opt, 'rb'))
            if C_opt_t.shape == (49, lambdas_.shape[0]):
                print('Optimal C_opt loaded from disk')
                env.C_opt = C_opt_t
            else:
                env.find_optimal_c_13(passes=1)
                pickle.dump(env.C_opt, open(path_c_opt, 'wb'))
        else:
            POPULATION = 100
            solver = PEPG(13 * env.N,  # number of model parameters
                          sigma_init=1,  # initial standard deviation
                          learning_rate=1e-3,  # learning rate for standard deviation
                          elite_ratio=0.15,
                          popsize=POPULATION,  # population size
                          average_baseline=True,  # set baseline to average of batch
                          weight_decay=0.00,  # weight decay coefficient
                          rank_fitness=False,  # use rank rather than fitness numbers
                          forget_best=False)  # don't keep the historical best solution)

            C = env.find_optimal_c_13_ES(solver, POPULATION, iters=1000)
            env.find_optimal_c_13(passes=1, initial=C)
            pickle.dump(env.C_opt, open(path_c_opt, 'wb'))


    # exit()

    C_opt = np.array(env.C_opt)

    # model = np.array([-3.02732536,  10.70271146,  -5.02678069, -18.22820558]) # np.random.rand(D)
    D = 10  # lambdas_.shape[0] * 2 + 1  #
    K = 1
    h_1_size = 45  # 100  # 20
    h_2_size = 5  # 50  # 10
    NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K

    weights_1_n = D * h_1_size
    bias_1_n = h_1_size
    weights_2_n = h_1_size * h_2_size
    bias_2_n = h_2_size
    weights_3_n = h_2_size * K
    bias_3_n = K
    # model = pretrain_network()

    s1 = time()

    NUM_TESTS = 1
    seeds = np.arange(NUM_TESTS)

    # FORCE_GLOBAL_ONE = True
    # solutions = np.tile(np.empty(NPARAMS), (NUM_TESTS, 1))
    #
    #
    # if varying_priority:
    #     if NUM_TESTS == 1:
    #         fitness_list = np.array([rollout_rep_when_updated((solutions[0], seeds[0]))])
    #     else:
    #         with ProcessPoolExecutor(max_workers=WORKERS) as executor:
    #             p = executor.map(rollout_rep_when_updated, zip(solutions, seeds))
    #             fitness_list = np.array(list(p))
    #
    #     when_first_node_updated = fitness_list[:, 1, 0]
    #     fitness_list = fitness_list[:, 0, 0]
    # else:
    #     if NUM_TESTS == 1:
    #         fitness_list = np.array([rollout_rep((solutions[0], seeds[0]))])
    #     else:
    #         with ProcessPoolExecutor(max_workers=WORKERS) as executor:
    #             p = executor.map(rollout_rep, zip(solutions, seeds))
    #             fitness_list = np.array(list(p))
    #
    # fitness_list = np.array(fitness_list).flatten()
    # print('This run yielded {} (percent {}). Seed {}.'.format(fitness_list.max() - fitness_list.min(), (fitness_list.max() - fitness_list.min()) / fitness_list.min(), seed))
    # print('Max sub-seed {}, Min sub-seed {}. SuperSeed {}. Mean {}.'.format(np.argmax(fitness_list), np.argmin(fitness_list), seed, fitness_list.mean()))
    # print('AT mean: {}. AT std: {}'.format(fitness_list.mean(), fitness_list.std()))
    # if varying_priority:
    #     print('First node was updated with N({}, {}) seconds'.format(when_first_node_updated.mean(), when_first_node_updated.std()))
    # print('Took {} seconds'.format(time() - s1))


    print('Selecting last generated model')
    exact_models = glob.glob('models/{} nodos*.p'.format(num_nodos))
    if len(exact_models) > 0:
        model_path = exact_models[0]
    else:
        models = glob.glob('models/model*.p')
        assert len(models) > 0
        models.sort(key=os.path.getmtime)
        model_path = models[-1]

    print('Found model to be loaded:', model_path)
    # model_path = 'models/50 nodos nuevo model-20180611-171206.p'
    model = pickle.load(open(model_path, 'rb'))

    assert model.shape[0] == NPARAMS

    FORCE_GLOBAL_ONE = False
    solutions = np.tile(model, (NUM_TESTS, 1))

    if varying_priority:
        if NUM_TESTS == 1:
            fitness_list_2 = np.array([rollout_rep_when_updated((solutions[0], seeds[0]))])
        else:
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                p = executor.map(rollout_rep_when_updated, zip(solutions, seeds))
                fitness_list_2 = np.array(list(p))

        when_first_node_updated = fitness_list_2[:, 1, 0]
        fitness_list_2 = fitness_list_2[:, 0, 0]
    else:
        if NUM_TESTS == 1:
            fitness_list_2 = np.array([rollout_rep((solutions[0], seeds[0]))])
        else:
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                p = executor.map(rollout_rep, zip(solutions, seeds))
                fitness_list_2 = np.array(list(p))

        fitness_list_2 = fitness_list_2.flatten()

    # print('This run yielded {} (percent {}). Seed {}.'.format(fitness_list_2.max() - fitness_list_2.min(), (fitness_list_2.max() - fitness_list_2.min()) / fitness_list_2.min(), seed))
    # print('Max sub-seed {}, Min sub-seed {}. SuperSeed {}. Mean {}.'.format(np.argmax(fitness_list_2), np.argmin(fitness_list_2), seed, fitness_list_2.mean()))
    print('ANN mean: {}. ANN std: {}'.format(fitness_list_2.mean(), fitness_list_2.std()))
    print('Took {} seconds'.format(time() - s1))
    # print("For {} runs. Total difference: {}. Percentual difference: {}".format(NUM_TESTS, fitness_list_2.mean() - fitness_list.mean(), (fitness_list_2.mean() - fitness_list.mean())/fitness_list.mean()))

    if varying_priority:
        print('First node was updated with N({}, {}) seconds'.format(when_first_node_updated.mean(), when_first_node_updated.std()))

    if NUM_TESTS == 1:
        fig1.savefig('accum_throughput_vs_time.png', dpi=200)
        fig1.savefig('accum_throughput_vs_time.eps')
        fig2.savefig('throughput_vs_time.png', dpi=200)
        fig2.savefig('throughput_vs_time.eps')
        fig3.savefig('updated_nodes_vs_time.png', dpi=200)
        fig3.savefig('updated_nodes_vs_time.eps')
        plt.show()

    exit()
