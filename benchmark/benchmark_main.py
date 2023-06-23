import json
import time
from datetime import datetime

import jax
import torch
from gpytorch.models import ExactGP
from jax import numpy as jnp
from jax.random import PRNGKeyArray
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ExponentialLR

from bbox import env, get_param, Function
from bbox import functions as fx
from bbox import wrappers as wx
from bbox._src.env.function_env import FunctionEnv
from bbox._src.types import Parameter
from agent import RandomAgent, qPES, qMES, qJES, qEI
from util import get_random_guess, tensor_to_jax_array, constrain

import sys
import gc

# get path to save files
path = ''
if len(sys.argv) > 1:
    # Access the argument
    path = sys.argv[1]

n_runs_per_agent = 3

### Run environment
key = jax.random.PRNGKey(25)

agents = {
    # 'random': RandomAgent,
    'qPES': qPES,
    # 'qMES': qMES,
    # 'qJES': qJES,
    # 'qEI': qEI,
}

Griewank = "Griewank"
Zakharov = "Zakharov"
Easom = "Easom"
Ackley = "Ackley"
DifferentPowers = "DifferentPowers"
Schwefel = "Schwefel"

testing_envs = {
    # Griewank: [{
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": 0.0,
    #     "n_iter": 80,
    # },
    #            {
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": .05,
    #     "n_iter": 80,
    # }, {
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": .10,
    #     "n_iter": 80,
    # }, {
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": .20,
    #     "n_iter": 80,
    # }, {
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": .40,
    #     "n_iter": 80,
    # }],
    Easom: [
    #     {
    #     "batch_size": 2,
    #     "dim": 2,
    #     "noise": .05,
    #     "n_iter": 80,
    # },
        {
        "batch_size": 2,
        "dim": 10,
        "noise": .05,
        "n_iter": 80,
    }, 
        {
        "batch_size": 2,
        "dim": 25,
        "noise": .05,
        "n_iter": 100,
    }, {
        "batch_size": 2,
        "dim": 50,
        "noise": .05,
        "n_iter": 80,
    },],
    Zakharov: [{
        "batch_size": 2,
        "dim": 2,
        "noise": .0,
        "n_iter": 80,
    }, {
        "batch_size": 5,
        "dim": 2,
        "noise": .0,
        "n_iter": 50,
    }, {
        "batch_size": 10,
        "dim": 2,
        "noise": .0,
        "n_iter": 40,
    }, {
        "batch_size": 25,
        "dim": 2,
        "noise": .0,
        "n_iter": 15,
    }],
    Ackley: [{
        "batch_size": 5,
        "dim": 2,
        "noise": .1,
        "n_iter": 80,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .1,
        "n_iter": 80,
    }, {
        "batch_size": 5,
        "dim": 25,
        "noise": .1,
        "n_iter": 80,
    }, {
        "batch_size": 5,
        "dim": 50,
        "noise": .1,
        "n_iter": 80,
    },],
    Schwefel: [{
        "batch_size": 2,
        "dim": 10,
        "noise": .2,
        "n_iter": 100,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .2,
        "n_iter": 70,
    }, {
        "batch_size": 10,
        "dim": 10,
        "noise": .2,
        "n_iter": 50,
    }, {
        "batch_size": 25,
        "dim": 10,
        "noise": .2,
        "n_iter": 20,
    }],
    DifferentPowers: [{
        "batch_size": 5,
        "dim": 10,
        "noise": .0,
        "n_iter": 50,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .05,
        "n_iter": 50,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .1,
        "n_iter": 50,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .2,
        "n_iter": 50,
    }, {
        "batch_size": 5,
        "dim": 10,
        "noise": .4,
        "n_iter": 50,
    }],
}


def create_function_env(function_name, n_dim, noise) -> (FunctionEnv, FunctionEnv):
    match function_name:
        case 'Griewank':
            base = fx.real.Griewank
        case 'Easom':
            base = fx.real.Easom
        case 'Ackley':
            base = fx.real.Ackley
        case 'DifferentPowers':
            base = fx.real.DifferentPowers
        case 'Schwefel':
            base = fx.real.Schwefel
        case 'Zakharov':
            base = fx.real.Zakharov
        case _:
            raise ValueError(f'Unknown function name: {function_name}')

    negative_wrapper: [Function] = [
        wx.FlipSign.partial()
    ]

    wrappers: [Function] = [
        wx.real.WhiteNoise.partial(stddev=noise)
    ]

    objective_function = env.as_bandit(
        base=base,
        wrappers=negative_wrapper,
        dummy_x=jnp.zeros(n_dim),
        register_optimum=True,
    )

    black_box_function = env.as_bandit(
        base=base,
        wrappers=negative_wrapper + wrappers,
        dummy_x=jnp.zeros(n_dim),
    )

    return objective_function, black_box_function


def get_noise(function_name, env_settings):
    match function_name:
        case 'Griewank':
            return env_settings['noise'] * 90 * env_settings['dim']
        case 'Easom':
            return env_settings['noise'] * 1
        case 'Ackley':
            return env_settings['noise'] * 21.5
        case 'DifferentPowers':
            return env_settings['noise'] * env_settings['dim']
        case 'Schwefel':
            # only used with dimension 50, function has range 25_000 in this case
            return env_settings['noise'] * 25_000
        case 'Zakharov':
            return env_settings['noise']
        case _:
            raise ValueError(f'Unknown function name: {function_name}')


def get_range(function_name, n_dim):
    match function_name:
        case 'Griewank':
            x_range = (-600, 600)
        case 'Easom':
            x_range = (-10, 10)
        case 'Ackley':
            x_range = (-32.768, 32.768)
        case 'DifferentPowers':
            x_range = (-1, 1)
        case 'Schwefel':
            x_range = (-500, 500)
        case 'Zakharov':
            x_range = (-5, 10)
        case _:
            raise ValueError(f'Unknown function name: {function_name}')


    x_range_per_dim = [x_range for _ in range(n_dim)]
    x_min = torch.tensor([x[0] for x in x_range_per_dim]).double()
    x_max = torch.tensor([x[1] for x in x_range_per_dim]).double()
    # x_min = torch.tensor([x[0] for x in x_range_per_dim])
    # x_max = torch.tensor([x[1] for x in x_range_per_dim])
    return torch.stack([x_min, x_max])


def find_argmax_mean(model: ExactGP, x_range: torch.Tensor) -> jax.Array:
    # find argmax of model(x).mean
    # get 10 random guesses
    x_guess = get_random_guess(x_range, 10_000)
    y_guess = model(x_guess).mean

    # find best guess
    x_best = x_guess[y_guess.argmax()]
    y_best = y_guess.max()

    # do LBFGS for 10 iterations
    for _ in range(10):
        x_init = get_random_guess(x_range, 1).requires_grad_(True)

        optimizer = LBFGS([x_init], lr=10., max_iter=50)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        def closure():
            optimizer.zero_grad()
            constrain(x_init, x_range)
            output = model(x_init)
            loss = -output.mean
            loss.backward()
            return loss

        for _ in range(20):
            optimizer.step(closure)
            scheduler.step()
        argmax = constrain(x_init, x_range)
        y_argmax = model(argmax).mean
        if x_best is None or y_argmax > y_best:
            x_best = argmax
            y_best = y_argmax

    return tensor_to_jax_array(x_best)


def write_to_file(data):
    file_name = 'test_results' + str(datetime.now()).replace(' ', 'T')
    file_name = file_name.split('.')[0]
    file_name = file_name.replace(':', '-')
    with open(path + file_name + '.json', mode='w') as json_file:
        json.dump(data, json_file)


def run_experiment(_key: PRNGKeyArray):
    # run for all agents
    # init dictionary for storing metrics
    data = {}

    for function, env_settings in testing_envs.items():
        print(f'Running experiment for {function}')
        # get environment settings
        for env_setting in env_settings:
            print(f'-- Running experiment for {function} with settings {env_setting}')
            # get environment settings
            n_dim = env_setting['dim']
            noise = get_noise(function, env_setting)
            x_range = get_range(function, n_dim)
            batch_size = env_setting['batch_size']
            n_iter = env_setting['n_iter']

            # objective function does not contain wrappers (ie: noise)
            objective_function, black_box_function = create_function_env(function, n_dim, noise)

            # get optimum value
            x = jax.random.uniform(_key, shape=(1, n_dim,))
            params = objective_function.init_fun(_key, x)
            opt_value = get_param(params, Parameter.OPTIMUM_VALUE)

            # run experiment for n_runs_per_agent iterations
            for agent_name, agent_clazz in agents.items():
                try:
                    print(f'---- Running experiment for {function} with settings {env_setting} and agent {agent_name}')

                    run_name = f'{function}_b:{batch_size}_d:{n_dim}_n:{env_setting["noise"]}'

                    if agent_name not in data:
                        data[agent_name] = {}
                    data[agent_name][run_name] = {}

                    agent = agent_clazz(agent_name, n_dim, x_range, batch_size)

                    # run experiment for n_runs_per_agent iterations
                    for run in range(n_runs_per_agent):
                        # reset state
                        true_state, true_step = objective_function.reset(_key)
                        state, step = black_box_function.reset(_key)
                        agent.reset()

                        # initialize and evaluate three random points in domain
                        x0 = tensor_to_jax_array(get_random_guess(x_range, 3))
                        _, true_step = jax.vmap(objective_function.step, in_axes=(None, 0))(true_state, x0)

                        _, step = jax.vmap(black_box_function.step, in_axes=(None, 0))(state, x0)
                        steps = [(x0, step)]

                        # initialize variables for metrics
                        #     cumulative regret: \sum_{over batch}[\opt_value - f(x)]
                        cumulative: [float] = []

                        #     simple regret:     \opt_value - f(\x_best_so_far)
                        best_value: float = jnp.max(true_step.reward).item()
                        simple: [float] = []

                        #     median immediate regret: median_{over batch}[\opt_value - f(x)]
                        immediate: [float] = []

                        #     inference regret: x_opt  - f(x_T)
                        #         x_T = argmax_{x in domain}\mean_T(x)   # mean of posterior
                        inference: [float] = []

                        # runtime
                        runtime: [float] = []

                        for _ in range(n_iter):
                            try:

                                _key, key_policy = jax.random.split(_key)

                                # Agent-Environment interaction
                                start = time.time()
                                action, model = agent.get_action(key_policy, steps[-1])
                                end = time.time()

                                _, true_step = jax.vmap(objective_function.step, in_axes=(None, 0))(true_state, action)

                                _, step = jax.vmap(black_box_function.step, in_axes=(None, 0))(state, action)
                                steps.append((action, step))

                                # update metrics
                                last_cumulative = cumulative[-1] if len(cumulative) > 0 else 0
                                cumulative.append(last_cumulative + jnp.sum(opt_value - true_step.reward).item())

                                round_best = jnp.max(true_step.reward).item()
                                if round_best > best_value:
                                    best_value = round_best
                                simple.append((opt_value - best_value).item())

                                immediate.append(jnp.median(jax.vmap(lambda x: opt_value - x)(true_step.reward)).item())

                                # if model is not None:
                                #     x_max_mean = find_argmax_mean(model, x_range)
                                #     inference.append((opt_value - objective_function.step(true_state, x_max_mean)[1].reward).item())

                                runtime.append(end - start)

                                gc.collect()

                            except Exception as e:
                                print(e)
                                break

                        data[agent_name][run_name][run] = {
                            # 'steps': steps,
                            'cumulative': cumulative,
                            'simple': simple,
                            'immediate': immediate,
                            'inference': inference,
                            'runtime': runtime,
                        }
                        write_to_file(data)
                except Exception as e:
                    print(e)
                    continue

    return data


data = run_experiment(key)
write_to_file(data)

exit(0)
