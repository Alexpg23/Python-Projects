import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3, DDPG #, ACKTR
import matplotlib.pyplot as plt
from itertools import product
from bs_hedging import bs_hedge

# basic variables of the universe 
s0 = 100.0  # Stock price at time = 0
strike = s0
sigma = 0.2  # Implied volatility
mu = 0.05
r = 0.0  # Risk-free rate
dividend = 0.0  # Continuous dividend yield

# cost variables
final_period_cost = True
epsilon = 0.01

# time variables
#n_calendar_bdays = 20
#dt_days = 1 # hedging frequency
dt_daycount = 252
#n_time_steps = int(n_calendar_bdays/dt_days) # should be product of 20 
#texp_start = 1/12 # how many months

dt_days_list = [1]#[1, 2, 3, 5] #[1]#
texp_start_list = [1/12] #[1/12, 3/12] #[1/12]#
launch_pnl_plot = True # True, False

# option type
payoff = 'european_call' # 'digital_call', 'european_call'

# simulation parameters
simulator = 'SV' # 'SV', 'GBM'

# model variables
model_type = 'SAC' # 'SAC', 'TD3', 'DDPG'
nn_structure = 'mlp'
std_const = 1.5
episodes = 5000
reward_type = 'HullCF_fpysim' # name of the model versions I have 'HullCF_fpysim', 'HullCF_future'
verbose = 0 # 0, 1

# parameters for loading the model
loading_path = 'european_call-SV-SAC-mlp-dt1-252-eps0.01-const1.5-n20 1653414854' # goes to models/reward_type/loading_path
model_version = '250-1000'

##########################################################################################

if reward_type == 'HullCF':
  from hedge_CF_env import HedgeEnv
elif reward_type == 'HullCF_future':
  from hedge_env_main import HedgeEnv
elif reward_type == 'HullCF_fpysim':
  from hedge_env_fpysim import HedgeEnv
elif reward_type == 'Artem':
  from hedge_env import HedgeEnv
elif reward_type == 'ArtemPnL':
  from hedge_PnL_env import HedgeEnv

print(f'Testing Results for model:{model_type}, simulator: {simulator}, NN-structure: {nn_structure}')

for texp_start, dt_days in list(product(texp_start_list, dt_days_list)):

  prices_array, final_wealth_array, start_option_array, deltas_array, rl_cost_history = [], [], [], [], []

  env = HedgeEnv(payoff=payoff,
                  s0=s0,
                  strike=strike,
                  sigma=sigma,
                  epsilon=epsilon,
                  risk_free=r,
                  dividend=dividend,
                  texp_start=texp_start, #n_time_steps=n_time_steps,
                  dt_days=dt_days,
                  dt_daycount=dt_daycount,
                  std_const=std_const,
                  initial_wealth=0.0,
                  final_period_cost=final_period_cost,
                  mu=mu,
                  simulator = simulator)
  env.reset()

  ##########################################################################################

  model_path = f'models/{reward_type}/{loading_path}/{model_version}'

  if model_type == 'SAC': 
    model = SAC.load(model_path, env=env)
  #elif model_type == 'ACKTR':
  #  model = ACKTR.load(model_path, env=env)
  elif model_type == 'TD3':
    model = TD3.load(model_path, env=env)
  elif model_type == 'DDPG': 
    model = DDPG.load(model_path, env=env)

  ##########################################################################################

  for ep in range(episodes):
      obs = env.reset()
      done = False
      while not done:
          action, _ = model.predict(obs)
          obs, reward, done, info = env.step(action)
      final_wealth, prices, start_option_value, rl_cost = env.get_info()
      delta_history = env.get_deltas()
      #prices_array.append(prices.tolist()[:-1]) 
      prices_array.append(prices) 
      final_wealth_array.append(final_wealth)
      start_option_array.append(start_option_value)
      rl_cost_history.append(rl_cost)
      deltas_array.append(delta_history)

  ##########################################################################################

  PnL_BS, cost_history_BS = bs_hedge(prices_array, 
                                      strike, 
                                      texp_start, 
                                      sigma, 
                                      r, 
                                      epsilon, 
                                      option_type=payoff, 
                                      leland_sigma=False, 
                                      dividend_yield=dividend, 
                                      final_period_cost=final_period_cost,
                                      verbose=verbose)

  ##########################################################################################

  start_option_val = start_option_array[0]

  #cost_pct_BS = [cost_history_BS[i]/start_option_val for i in range(len(cost_history_BS))]
  #cost_pct_RL = [rl_cost_history[i]/start_option_val for i in range(len(rl_cost_history))]

  bs_row = [-np.mean(PnL_BS)/start_option_val, np.std(PnL_BS)/start_option_val] #, np.mean(cost_pct_BS), np.std(cost_pct_BS)]
  rl_row = [-np.mean(final_wealth_array)/start_option_val, np.std(final_wealth_array)/start_option_val] #, np.mean(cost_pct_RL), np.std(cost_pct_RL)]
  Y0_improvement = np.mean([(-i+j)/-i for i, j in zip(PnL_BS, final_wealth_array)])

  print(f'Results for dt{dt_days}, expiry: {texp_start*12}')
  print(pd.DataFrame(columns=['Mean PnL %', 'Std PnL %'], index=['BS','RL'], data=[bs_row, rl_row])) #print(pd.DataFrame(columns=['Mean PnL %', 'Std PnL %', 'Mean Cost %', 'Std Cost %'], index=['BS','RL'], data=[bs_row, rl_row]))
  print('Improvement case-on-case:', Y0_improvement)

  ##########################################################################################

  if launch_pnl_plot:
    bar1 = PnL_BS
    bar2 = final_wealth_array #+ price_BS[0][0]
    # Plot Black-Scholes PnL and Deep Hedging PnL (with BS_price charged on both).
    fig_PnL = plt.figure(dpi= 125, facecolor='w')
    fig_PnL.suptitle("Black-Scholes PnL vs Deep Hedging PnL \n", fontweight="bold")
    ax = fig_PnL.add_subplot()
    ax.set_title(f"Option = {payoff}, Model = {model_type}, Epsilon = {epsilon}, dt = {dt_days}", fontsize=8)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.hist((bar1,bar2), bins=30, er)
    ax.legend()
    plt.show()

  ##########################################################################################

    fig_deltas = plt.figure(dpi= 125, facecolor='w')
    fig_deltas.suptitle("RL Hedging Policy Deltas \n", fontweight="bold")
    ax = fig_deltas.add_subplot()
    ax.set_title(f"Option = {payoff}, Model = {model_type}, Epsilon = {epsilon}, dt = {dt_days}", fontsize=8)
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Delta")
    ax.scatter(prices_array, deltas_array)
    plt.show()

##########################################################################################