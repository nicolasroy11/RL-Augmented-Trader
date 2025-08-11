# RL-Augmented-Trader

**A real-time, deep reinforcement learning trading bot using live Binance 5-second tick data, technical indicators, and a custom PyTorch policy gradient agent to make buy/hold/sell decisions.**

This project is a hands-on experiment in building an RL-powered trading system from scratch — no high-level frameworks like Stable-Baselines or FreqTrade. Just raw PyTorch, real-time data, and a simple, focused environment to simulate market PnL.

---

## Why This Project?

This experiment is meant to enhance an existing, successful, rules-based trading bot with RL and NN features. This is part of my portfolio to demonstrate **end-to-end applied machine learning** in a real financial setting. It combines:

- Low-latency live data ingestion  
- Feature engineering using technical indicators  
- A from-scratch trading environment  
- A custom RL agent (initially a policy gradient model)

The goal? **Maximize real trading performance** — and push this agent to a level that would be compelling to hedge fund researchers and quant teams.

---

Contact: nicolasroy11@gmail.com



## Features

✅ Real-time 5s data ingestion from Binance  
✅ DuckDB-based tick storage for fast analytics  
✅ Custom PyTorch RL agent (Policy Gradient)  
✅ Live environment with simulated trading logic  
✅ Normalized technical indicators: RSI, EMA, MACD, Bollinger bands  
✅ Finplot-based data visualization  
✅ Training insights: logits, action probabilities, PnL tracking    
✅ Django apps for visualization and eventually meaningful interaction  


## Background: The Lowrider

In my years of creating rules-based trading automations from absolute scratch, one agent has been consistently successful: the Lowrider. This agent's basic set of principles read like the age old adage of buy low, sell high. Here, the distinction happens in the methodology. The agent enters a trade cycle by buying a small amount without regards to market conditions and only sells when a profit is reached. But what happens if the market moves against its position? It buys again at the new, lower price, in what is commonly referred to as a DCA (dollar-cost averaging) maneuver. The lower the price goes, the higher the lot size being bought, and thus the faster the averaging catches up to the break-even point in such a way that your average upward market correction in a downward trend is enough to drive the average position price above break even. At this point, the agent sells. Selling a trade cycle at a loss is not an option - you either buy more or sell at a profit. In this way, the negative trade cycle outcome is mechanically impossible.

I have been running this agent on a cloud instance with roughly this logic for just over a year and a half with fairly consistent returns annualizing at ~20-30% of initial wallet size, barring any hiccups and interruptions.

The problem, as some may have guessed, is that in a steep downtrend, the occasional upward correction may not be enough to salvage a failing trade cycle, and the agent may continue to buy and buy and eventually run out of funds. I have seen it happen, and although it always recoverd naturally, it is unwise to rely on the recovery of the market alone.

One solution was to increase what I call the reluctance factor, which is a very primitive coefficient used to determine the maximum amount that can be spent on a single trade at any point in a trade cycle. Simply put, that amount will be 

    highest single buy size = wallet size/reluctance factor.

This crude method at least puts a clamp on runaway buying, the tradeoff being that in an upward trending market, the agent will only be making tiny profits, since it is saving its larger buys for lower lows.

There are many details involved in the sizing of lots, some of which actually involved using an integral to figure out, but that is outside the scope of this description, and a patent may be pending in the near future. For this reason, the Lowrider repo has to be kept private until this experiment concludes.


## The RL Angle: Adaptive Lot Sizing

The contents of this repository will be an exploration of ways to better size lots given the current market conditions and adapt the Lowrider to use this to profit maximally in both upward and downward trending markets.

The concept is rather simple: take in a time window of recent market observations, feed them to a neural network and output the probablities for three states: buy, sell, or hold. The buy probability can then be used directly on the lot calculation, leading to larger long-term gains.


## This Repository

### Repo Structure TL;DR

- `db/` — Data scraping, storage, and visualization modules. Collects and stores Binance 5-second tick data with technical indicators (RSI, EMA, MACD).  
- `RL/` — Reinforcement learning agents and training scripts. Currently includes a stochastic policy gradient agent that learns over tick windows.
- `services/` — Django services that will be used to make this project interactive over the web.

---

The db folder contains a module that strictly concerns itself with the scraping, storing, and visualization of raw data. Since the nature of volatile assets presents tiny opportunities to profitably buy and sell at any moment regardless of wider trend, I opted to walk away from querying trading platform APIs for minutely OHLCV data, favoring the storage of sub-minute instantaneous tick and technical indicator (RSI, MACD, EMA in various lengths,) exactly as they would appear in a live scenario. A quick snapshot of current conditions. The interval is adjustable, and the default is 5-second intervals.

The RL folder houses the logic that will produce our model. In a bid to decouple this approach from my own trading biases and observations, I opted to start the exploration off with a purely stochastic run. Just collect a large amount of ticks and indicators and let the simplest feed-forward network buy and sell at random while keeping track of reward and PnL states. From observing the output vector change in real time as the network works through the data, it is visually evident that converging values are reached, confirming the model's confidence in its increasing experience.

The services folder is the home folder of all things Django service. I tend to set up Django services in a particular way as to make adding endpoints and functionality semantic and easy and using decorators in the same way they are used in .NET where type safety and return data structures are documented and Swagger-ready. This set of services will allow for theongoing processes of agent automation to be modified and occasionally course-corrected in real time if need be.


## Getting Started 🚀

### 1. Clone the repo
```bash
git clone https://github.com/nicolasroy11/RL-Augmented-Trader.git
```

### Navigate into the repo
```bash
cd RL-Augmented-Trader
```

### Install the requirements
```bash
pip install -r requirements.txt
```

### Run the data storage for any desired amount of time
```bash
export PYTHONPATH=.
python db/datastore.py
```

### Once you have completed the above step, you can run a quick visualization of what you've collected
```bash
python db/visualization.py
```

### To run the stochastic training experiment, use the following command:
```bash
python RL/playground/stochastic/run.py
```


## Roadmap / TODO

- [ ] Integrate PPO agent for more stable learning
- [ ] Add Sharpe ratio evaluations and other standard performance markers
- [ ] Improve model performance tracking with additional tooling
- [ ] Publish blog write-up on training insights
- [ ] Once a successful model is generated, integrate it into the existing Lowrider code
- [ ] Broadcast buy/sell signals for consumption by third party traders (Zignaly?)
- [ ] Include LSTM/GRU networks in the RL process to strengthen regime awareness
- [ ] Expand Django apps to expose the process to external parties


