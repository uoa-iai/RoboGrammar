'''
heuristic_search_algo_1.py

Implement the first algorithm in the pdf: overleaf.com/project/5ea0b6ae56ec33000137a41c

Learn a heuristic function V(s) for each universal design to predict the best reward under that node in the searching tree.

1. Search strategy is epsilon greedy.
2. Update the target value for the ancestors of the explored design
3. train V to fit target value
'''

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')      # sets base directory one level higher than 'graph_learning' (@ 'examples\')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))

# import python packages
import argparse
import time
import random
import pickle
import csv
import ast
import numpy as np
from copy import deepcopy
import torch
from torch import optim
import torch.nn.functional as F
from torch_geometric.data.data import Data

# import our own packages
from arguments import get_parser
from utils import solve_argv_conflict
from common import *
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals
import tasks
import pyrobotdesign as rd
from RobotGrammarEnv import RobotGrammarEnv
from Net import Net
from Preprocessor import Preprocessor
from states_pool import StatesPool

# predict without grad
def predict(V, state):
    global preprocessor
    adj_matrix_np, features_np, _ = preprocessor.preprocess(state)  # convert state into adj_matrix and feature vector
    masks_np = np.full(len(features_np), True)
    with torch.no_grad():
        features = torch.tensor(features_np).unsqueeze(0)           # unsqueeze simply returns a tensor with dimension_size=1, at index 0
        adj_matrix = torch.tensor(adj_matrix_np).unsqueeze(0)
        masks = torch.tensor(masks_np).unsqueeze(0)
        output, _, _ = V(features, adj_matrix, masks)               # carry out value prediction using V (GNN)
        return output.item()

def predict_batch(V, states):
    global preprocessor
    adj_matrix_np, features_np, masks_np = [], [], []
    max_nodes = 0
    for state in states:                                            # for each state in list of robot states
        adj_matrix, features, _ = preprocessor.preprocess(state)    # prepare robot graph/state for value prediction
        max_nodes = max(max_nodes, len(features))
        adj_matrix_np.append(adj_matrix)                            # convert to numpy matrix for tensor
        features_np.append(features)                                # convert to numpy array for tensor

    for i in range(len(states)):
        # pad matrix/features to make all matrices equal lengths
        adj_matrix_np[i], features_np[i], masks = \
            preprocessor.pad_graph(adj_matrix_np[i], features_np[i], max_nodes)
        masks_np.append(masks)

    with torch.no_grad():
        # convert numpy matrices to tensor matrices for CPU artihmetics
        adj_matrix = torch.tensor(adj_matrix_np)
        features = torch.tensor(features_np)
        masks = torch.tensor(masks_np)
        output, _, _ = V(features, adj_matrix, masks)               # carry out value prediction using tensors
    return output[:, 0].detach().numpy()

def select_action(env, V, state, eps):
    available_actions = env.get_available_actions(state)            # get set of available actions (rules) for current state
    if len(available_actions) == 0:
        return None, None
    sample = random.random()
    step_type = ""
    if sample > eps:        # choose optimal action to allow convergence
        next_states = []
        for action in available_actions:
            next_states.append(env.transite(state, action))         # carry out action and add updated state to list of next_states
        values = predict_batch(V, next_states)                      # get value prediction from batch of states
        best_action = available_actions[np.argmax(values)]          # keep the best predicted value, get corresponding action
        step_type = 'optimal'
    else:                   # choose random action to allow exploration
        best_action = available_actions[random.randrange(len(available_actions))]   # action is chosen randomly from set of available actions
        step_type = 'random'
    
    return best_action, step_type

def update_Vhat(args, V_hat, state_seq, reward):
    for state in state_seq:
        state_hash_key = hash(state)                                # convert state to a hash (foe quick lookup table)
        if not (state_hash_key in V_hat):                           # if not exist, create an entry
            V_hat[state_hash_key] = -np.inf
        V_hat[state_hash_key] = max(V_hat[state_hash_key], reward)  # update reward with max of current or previous reward

def update_states_pool(states_pool, state_seq, states_set):
    for state in state_seq:
        state_hash_key = hash(state)
        if not (state_hash_key in states_set):
            states_pool.push(state)
            states_set.add(state_hash_key)

def apply_rules(state, actions, env):
    cur_state = state
    state_seq = [cur_state]
    for action in actions:
        cur_state = env.transite(cur_state, action)
        state_seq.append(cur_state)
    return cur_state, state_seq

def show_all_actions(env, V, V_hat, state):
    available_actions = env.get_available_actions(state)
    #print('V(state) = ', predict(V, state), ', V_hat(state) = ', V_hat[hash(state)])
    for action in available_actions:
        next_state = env.transite(state, action)
        print('action = ', action, ', V = ', predict(V, next_state), ', V_hat = ', V_hat[hash(next_state)] if hash(next_state) in V_hat else -1.0)

def search_algo(args):
    ## BEGIN INITIALISE ##

    # iniailize random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    
    # initialize/load task
    # sets the default simulation and Props/objects in the environment
    # also sets maximum reasonable result (result_bound = 10.0)
    task_class = getattr(tasks, args.task)
    if args.no_noise:
        task = task_class(force_std = 0.0, torque_std = 0.0)
    else:
        task = task_class()
    graphs = rd.load_graphs(args.grammar_file)                      # load subgraphs into a vector of Graphs
    rules = [rd.create_rule_from_graph(g) for g in graphs]          # loads individual rules from vector of Graphs

    # initialize preprocessor
    # Find all possible (unique) link labels, so they can be one-hot encoded (binarization of categorical values)
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)                # a set only holds unique values so adding same thing doesn't save duplicates
    all_labels = sorted(list(all_labels))                           # sorts the set alphabetically
    
    # TODO: use 80 to fit the input of trained MPC GNN, use args.depth * 3 later for real mpc
    max_nodes = args.max_nodes

    # creates a global preprocessor object with labels sets from the set of link_labels
    global preprocessor
    # preprocessor = Preprocessor(max_nodes = max_nodes, all_labels = all_labels)
    preprocessor = Preprocessor(all_labels = all_labels)

    # initialize the environment
    env = RobotGrammarEnv(task, rules, seed = args.seed, mpc_num_processes = args.mpc_num_processes)

    # initialize Value function
    device = 'cpu'
    state = env.reset()                                             # reset state to get initial single node graph
    sample_adj_matrix, sample_features, sample_masks = preprocessor.preprocess(state)       # initialise defualt adjacency matrix, features matrix, and mask
    num_features = sample_features.shape[1]                         # get number of features
    # load value function as a graph neural network which computes on 'cpu'
    V = Net(max_nodes = max_nodes, num_channels = num_features, num_outputs = 1).to(device)

    # load pretrained V function if arg is given
    if args.load_V_path is not None:
        V.load_state_dict(torch.load(args.load_V_path))
        print_info('Loaded pretrained V function from {}'.format(args.load_V_path))
    
    # initialize target V_hat look up table
    V_hat = dict()

    # load pretrained V_hat if arg is given
    if args.load_Vhat_path is not None:
        V_hat_fp = open(args.load_Vhat_path, 'rb')                  # open pre-trained file for V_hat
        V_hat = pickle.load(V_hat_fp)                               # unpickle (de-serialise) the V_hat
        V_hat_fp.close()
        print_info('Loaded pretrained Vhat from {}'.format(args.load_Vhat_path))

    # train the V function else test the V function (commented code below)
    if not args.test:
        # initialize save folders and files
        fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
        fp_log.close()
        fp_eval = open(os.path.join(args.save_dir, 'eval.txt'), 'w')
        fp_eval.close()
        design_csv_path = os.path.join(args.save_dir, 'designs.csv')
        fp_csv = open(design_csv_path, 'w')
        fieldnames = ['rule_seq', 'reward', 'opt_seed']
        writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
        writer.writeheader()
        fp_csv.close()

        # initialize the global Adam optimizer (lr = 0.0001)
        global optimizer
        optimizer = torch.optim.Adam(V.parameters(), lr = args.lr)

        # initialize the seen states pool (capacity = 10,000,000)
        states_pool = StatesPool(capacity = args.states_pool_capacity)
        states_set = set()

        # explored designs
        designs = []
        design_rewards = []
        design_opt_seeds = []

        # initialize best design rule sequence
        best_design, best_reward = None, -np.inf

        # reward history
        epoch_rew_his = []
        last_checkpoint = -1

        # recording time
        t_sample_sum = 0.               # NOTICE FLOAT...
        num_samples_interval = 0
        max_seen_nodes = 0

        # record the count for invalid samples
        no_action_samples, step_exceeded_samples, self_collision_samples = 0, 0, 0

        # initialize stats variables
        num_invalid_samples, num_valid_samples = 0, 0
        repeated_cnt = 0

        # record prediction error
        prediction_error = []
        
        ## BEGIN ITERATIONS ##
        for epoch in range(args.num_iterations):
            t_start = time.time()

            V.eval()        # Sets V function into evaluation mode (rather than training mode)

            # update eps and eps_sample (either linearly or exponentially)
            # eps is the exploration parameter (as it decreases we explore less and want to converge to optimal design)
            if args.eps_schedule == 'linear-decay':
                eps = args.eps_start + epoch / args.num_iterations * (args.eps_end - args.eps_start)
            elif args.eps_schedule == 'exp-decay':
                eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-1.0 * epoch / args.num_iterations / args.eps_decay)

            if args.eps_sample_schedule == 'linear-decay':
                eps_sample = args.eps_sample_start + epoch / args.num_iterations * (args.eps_sample_end - args.eps_sample_start)
            elif args.eps_sample_schedule == 'exp-decay':
                eps_sample = args.eps_sample_end + (args.eps_sample_start - args.eps_sample_end) * np.exp(-1.0 * epoch / args.num_iterations / args.eps_sample_decay)

            t_sample, t_update, t_mpc, t_opt = 0, 0, 0, 0

            # reset selected designs, rules, state sequences and rule sequences
            selected_design, selected_reward = None, -np.inf
            selected_state_seq, selected_rule_seq = None, None

            p = random.random()
            if p < eps_sample:      # as eps_sample gets smaller; more likely to sample multiple designs (num_samples = args.num_samples)
                num_samples = 1
            else:
                num_samples = args.num_samples

            ### DESIGN PHASE ###
            # use e-greedy to sample a design within maximum #steps.
            for _ in range(num_samples):
                valid = False
                while not valid:                                    # keep looping until you get a valid design
                    t0 = time.time()

                    state = env.reset()                             # reset environment; state is initial state (single node graph)
                    rule_seq = []                                   # rule sequence is reset
                    state_seq = [state]                             # store initial state in state sequence
                    no_action_flag = False                          # reset flag which shows no more actions available to choose from
                    for _ in range(args.depth):                     # rule sequence to be length of depth
                        action, step_type = select_action(env, V, state, eps)       # choose an action whose label matches the required_label and is applicable
                        if action is None:
                            no_action_flag = True                   # can't find any more actions so stop rule search
                            break
                        rule_seq.append(action)                     # append that action to rule sequence
                        next_state = env.transite(state, action)    # carry out action on current state and save updated state
                        state_seq.append(next_state)                # add updated state to state sequence list
                        state = next_state
                        #show_all_actions(env, V, V_hat, state)      # print all available actions
                        if not has_nonterminals(state):             # if completed graph (terminal states only) then stop
                            break
                    
                    valid = env.is_valid(state)                     # check if design is valid (terminal states only and no self_collisions)

                    t_sample += time.time() - t0

                    t0 = time.time()

                    if not valid:
                        # update the invalid sample's count
                        if no_action_flag:
                            no_action_samples += 1
                        elif has_nonterminals(state):
                            step_exceeded_samples += 1
                        else:
                            self_collision_samples += 1
                        num_invalid_samples += 1
                    else:
                        num_valid_samples += 1
                    
                    num_samples_interval += 1

                    t_update += time.time() - t0

                predicted_value = predict(V, state)                                 # predict the reward with current valid robot design
                if predicted_value > selected_reward:                               # if current design better than previous seen designs...
                    selected_design, selected_reward = state, predicted_value       # save the design and predicted reward
                    selected_rule_seq, selected_state_seq = rule_seq, state_seq     # save the rule and state sequence (partial designs)

            ### EVALUATION PHASE ###
            t0 = time.time()                                        # once all sample designs are generated reset t0 time

            repeated = False
            if hash(selected_design) in V_hat:                      # check if design has been saved before
                repeated = True
                repeated_cnt += 1

            reward, best_seed = -np.inf, None
            
            for _ in range(args.num_eval):                          # evaluate the design multiple times (different seeds)
                _, rew = env.get_reward(selected_design)            # get reward from the selected design (each iter reward can vary)
                if rew > reward:                                    # if current design's reward is larger then previous evals
                    reward, best_seed = rew, env.last_opt_seed      # then store reward as well as the seed to that design

            t_mpc += time.time() - t0

            # save the design, its reward, and seed in their respective lists
            designs.append(selected_rule_seq)
            design_rewards.append(reward)
            design_opt_seeds.append(best_seed)

            # update best design
            if reward > best_reward:
                best_design, best_reward = selected_rule_seq, reward
                print_info('new best: reward = {:.4f}, predicted reward = {:.4f}, num_samples = {}'.format(reward, selected_reward, num_samples))

            t0 = time.time()

            # update V_hat for the valid design
            update_Vhat(args, V_hat, selected_state_seq, reward)

            # update states pool for the valid design
            update_states_pool(states_pool, selected_state_seq, states_set)

            t_update += time.time() - t0

            t0 = time.time()

            ### LEARNING PHASE ###
            # optimize
            V.train()               # set the GNN (V) to training mode
            total_loss = 0.0
            for _ in range(args.opt_iter):
                # get a random sample of min(total_states, 32) states from pool of states
                minibatch = states_pool.sample(min(len(states_pool), args.batch_size))
                
                train_adj_matrix, train_features, train_masks, train_reward = [], [], [], []
                max_nodes = 0
                for robot_graph in minibatch:                                       # use individual robots from pool for training
                    hash_key = hash(robot_graph)
                    target_reward = V_hat[hash_key]
                    adj_matrix, features, _ = preprocessor.preprocess(robot_graph)  # create robot's adjacency matrix, feature matrix
                    max_nodes = max(max_nodes, len(features))
                    train_adj_matrix.append(adj_matrix)
                    train_features.append(features)
                    train_reward.append(target_reward)

                max_seen_nodes = max(max_seen_nodes, max_nodes)

                for i in range(len(minibatch)):                                     # pad the matrices to all be equal
                    train_adj_matrix[i], train_features[i], masks = \
                        preprocessor.pad_graph(train_adj_matrix[i], train_features[i], max_nodes)
                    train_masks.append(masks)

                train_adj_matrix_torch = torch.tensor(train_adj_matrix)             # tensor for the training adjacency matrix
                train_features_torch = torch.tensor(train_features)                 # tensor for the training features
                train_masks_torch = torch.tensor(train_masks)                       # tensor for the training masks
                train_reward_torch = torch.tensor(train_reward)                     # tensor for the training reward 
                
                optimizer.zero_grad()                                               # sets gradient of all optimized tensors to 0
                # use the tensors to train V function and get output and losses
                output, loss_link, loss_entropy = V(train_features_torch, train_adj_matrix_torch, train_masks_torch)
                loss = F.mse_loss(output[:, 0], train_reward_torch)                 # computes mse between output and best reward for design
                loss.backward()                                                     # computes gradient of current tensor w.r.t graph leaves
                total_loss += loss.item()                                           # gets scalar value held in loss
                optimizer.step()                                                    # perform single optimization step (update parameters of model V)

            t_opt += time.time() - t0

            t_end = time.time()

            t_sample_sum += t_sample

            # logging
            if (epoch + 1) % args.log_interval == 0 or epoch + 1 == args.num_iterations:
                iter_save_dir = os.path.join(args.save_dir, '{}'.format(epoch + 1))
                os.makedirs(os.path.join(iter_save_dir), exist_ok = True)
                # save model
                save_path = os.path.join(iter_save_dir, 'V_model.pt')
                torch.save(V.state_dict(), save_path)
                # save V_hat
                save_path = os.path.join(iter_save_dir, 'V_hat')
                fp = open(save_path, 'wb')
                pickle.dump(V_hat, fp)
                fp.close()

            # save explored design and its reward to file
            fp_csv = open(design_csv_path, 'a')
            fieldnames = ['rule_seq', 'reward', 'opt_seed']
            writer = csv.DictWriter(fp_csv, fieldnames=fieldnames)
            for i in range(last_checkpoint + 1, len(designs)):
                writer.writerow({'rule_seq': str(designs[i]), 'reward': design_rewards[i], 'opt_seed': design_opt_seeds[i]})
            last_checkpoint = len(designs) - 1
            fp_csv.close()

            epoch_rew_his.append(reward)

            avg_loss = total_loss / args.opt_iter
            len_his = min(len(epoch_rew_his), 30)
            avg_reward = np.sum(epoch_rew_his[-len_his:]) / len_his
            prediction_error.append(np.abs(selected_reward - reward))
            avg_prediction_error = np.sum(prediction_error[-len_his:]) / len_his

            if repeated:
                print_white('Epoch {:4}: T_sample = {:5.2f}, T_mpc = {:5.2f}, T_opt = {:5.2f}, eps = {:5.3f}, eps_sample = {:5.3f}, #samples = {:2}, training loss = {:7.4f}, avg_pred_error = {:6.4f}, predicted_reward = {:6.4f}, reward = {:6.4f}, last 30 epoch reward = {:6.4f}, best reward = {:6.4f}'.format(\
                    epoch, t_sample, t_mpc, t_opt, eps, eps_sample, num_samples, \
                    avg_loss, avg_prediction_error, selected_reward, reward, avg_reward, best_reward))
            else:
                print_warning('Epoch {:4}: T_sample = {:5.2f}, T_mpc = {:5.2f}, T_opt = {:5.2f}, eps = {:5.3f}, eps_sample = {:5.3f}, #samples = {:2}, training loss = {:7.4f}, avg_pred_error = {:6.4f}, predicted_reward = {:6.4f}, reward = {:6.4f}, last 30 epoch reward = {:6.4f}, best reward = {:6.4f}'.format(\
                    epoch, t_sample, t_mpc, t_opt, eps, eps_sample, num_samples, \
                    avg_loss, avg_prediction_error, selected_reward, reward, avg_reward, best_reward))

            fp_log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
            fp_log.write('eps = {:.4f}, eps_sample = {:.4f}, num_samples = {}, T_sample = {:4f}, T_update = {:4f}, T_mpc = {:.4f}, T_opt = {:.4f}, loss = {:.4f}, predicted_reward = {:.4f}, reward = {:.4f}, avg_reward = {:.4f}\n'.format(\
                eps, eps_sample, num_samples, t_sample, t_update, t_mpc, t_opt, avg_loss, selected_reward, reward, avg_reward))
            fp_log.close()

            if (epoch + 1) % args.log_interval == 0:
                print_info('Avg sampling time for last {} epoch: {:.4f} second / sample'.format(args.log_interval, t_sample_sum / num_samples_interval))
                t_sample_sum = 0.
                num_samples_interval = 0
                print_info('max seen nodes = {}'.format(max_seen_nodes))
                print_info('size of states_pool = {}'.format(len(states_pool)))
                print_info('#valid samples = {}, #invalid samples = {}, #valid / #invalid = {}'.format(num_valid_samples, num_invalid_samples, num_valid_samples / num_invalid_samples if num_invalid_samples > 0 else 10000.0))
                print_info('Invalid samples: #no_action_samples = {}, #step_exceeded_samples = {}, #self_collision_samples = {}'.format(no_action_samples, step_exceeded_samples, self_collision_samples))
                print_info('repeated rate = {}'.format(repeated_cnt / (epoch + 1)))

        save_path = os.path.join(args.save_dir, 'model_state_dict_final.pt')
        torch.save(V.state_dict(), save_path)
    #else:
    #    import IPython
    #    IPython.embed()

    #    # test
    #    V.eval()
    #    print('Start testing')
    #    test_epoch = 10
    #    y0 = []
    #    y1 = []
    #    x = []
    #    for ii in range(0, 6):
    #        eps = 1.0 - 0.2 * ii

    #        print('------------------------------------------')
    #        print('eps = ', eps)

    #        reward_sum = 0.
    #        best_reward = -np.inf
    #        for epoch in range(test_epoch):
    #            t_sample = 0.
    #            valid = False
    #            trials = 0
    #            while not valid:
    #                trials += 1

    #                t0 = time.time()

    #                state = env.reset()
    #                rule_seq = []
    #                state_seq = [state]
    #                no_action_flag = False
    #                for _ in range(args.depth):
    #                    action, step_type = select_action(env, V, state, eps)
    #                    if action is None:
    #                        no_action_flag = True
    #                        break
    #                    rule_seq.append(action)
    #                    next_state = env.transite(state, action)
    #                    state_seq.append(next_state)
    #                    state = next_state
    #                    if not has_nonterminals(state):
    #                        break

    #                valid = env.is_valid(state)

    #                t_sample += time.time() - t0

    #            sys.stdout.write('\rrunning mpc')
    #            sys.stdout.flush()

    #            t0 = time.time()
    #            _, reward = env.get_reward(state)
    #            t_mpc = time.time() - t0

    #            reward_sum += reward
    #            best_reward = max(best_reward, reward)
    #            sys.stdout.write('\rdesign {}: reward = {}, trials = {}, t_mpc = {:.2f}, t_sample = {:.2f}'.format(epoch, reward, trials, t_mpc, t_sample))
    #            sys.stdout.write('\n')
    #            sys.stdout.flush()

    #        print('test avg reward = ', reward_sum / test_epoch)
    #        print('best reward found = ', best_reward)
    #        x.append(eps)
    #        y0.append(reward_sum / test_epoch)
    #        y1.append(best_reward)

    #    import matplotlib.pyplot as plt
    #    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    #    ax[0].plot(x, y0)
    #    ax[0].set_title('Avg Reward')
    #    ax[0].set_xlabel('eps')
    #    ax[0].set_ylabel('reward')

    #    ax[1].plot(x, y1)
    #    ax[0].set_title('Best Reward')
    #    ax[0].set_xlabel('eps')
    #    ax[0].set_ylabel('reward')

    #    plt.show()

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--task', 'FlatTerrainTask',
                 '--grammar-file', '../../data/designs/grammar_apr30.dot',
                 '--num-iterations', '2000',
                 '--mpc-num-processes', '32',
                 '--lr', '1e-4',
                 '--eps-start', '1.0',
                 '--eps-end', '0.1',
                 '--eps-decay', '0.3',
                 '--eps-schedule', 'exp-decay',
                 '--eps-sample-start', '1.0',
                 '--eps-sample-end', '0.1',
                 '--eps-sample-decay', '0.3',
                 '--eps-sample-schedule', 'exp-decay',
                 '--num-samples', '16', 
                 '--opt-iter', '25', 
                 '--batch-size', '32',
                 '--states-pool-capacity', '10_000_000',
                 '--depth', '40',
                 '--max-nodes', '80',
                 '--save-dir', './trained_models/',
                 '--log-interval', '100',
                 '--eval-interval', '1000',
                 '--max-trials', '1000',
                 '--num-eval', '1',
                 '--no-noise']

    # solve argument conflicts between input and default values
    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    # create directory for generated design
    if not args.test:
        args.save_dir = os.path.join(args.save_dir, args.task, get_time_stamp())
        try:
            os.makedirs(args.save_dir, exist_ok = True)
        except OSError:
            pass
        
        # save argument parameters to args.txt file
        fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
        fp.write(str(args_list + sys.argv[1:]))
        fp.close()

    search_algo(args)           # run the heuristic search algorithm
