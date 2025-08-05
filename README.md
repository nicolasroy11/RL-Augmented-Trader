# RL-Augmented-Trader
## Traditional rules-based automations strategies supplemented with reinforcement learning 

### Background

In my years of creating rules-based trading automations, one agent has been consistently successful: the Lowrider. This agent's basic set of principles read like the age old adage of buy low, sell high. Here, the distinction happens in the methodology. The agent buys at any moment without regards to market conditions and only sells when a profit is reached. But what happens if the market moves against its position? It buys again at the new, lower price, in what is commonly referred to as a DCA (dollar-cost averaging) maneuver. The lower the price goes, the higher the lot size being bought, and thus the faster the averaging catches up to the break-even point in such a way that your average upward market correction in a downward trend is enough to drive the average position price above break even. At this point, the agent sells. In this way, the negative trade outcome is mechanically  impossible.

I have been running this agent on a cloud instance with roughly this logic for just over a year and a half with fairly consistent returns annualizing at ~20-30% of initial wallet size, barring any hiccups and interruptions.

The problem, as some may have guessed, is that in a steep downtrend, the occasional upward correction may not be enough to salvage an ongoing trade, and the agent may continue to buy and buy and eventually run out of funds. I have seen it happen, and although it always recoverd naturally, it is unwise to rely on luck alone.

One solution was to increase what I call the reluctance factor, which is a very primitive coefficient used to determine the maximum amount that can be spent on a single trade at any point in a trade cycle. Simply put, that amount will be 

wallet size/reluctance factor.

This crude method at least puts a clamp on runaway buying, the tradeoff being that in an upward trending market, the agent will only be making tiny profits, since it is saving its larger buys for lower lows.

There are many details involved in the sizing of lots, some of which actually involved using an integral to figure out, but that is outside the scope of this description, and a patent may be pending in the near future.


### The RL Angle

The contents of this repository will be an exploration of ways to better size lots given the current market conditions and adapt the Lowrider to use this to profit maximally in both upward and downward trending markets.

The concept is rather simple: take in a time window of recent market observations, feed them to a neural network and output the probablities for three states: buy, sell, or hold. The buy probablity can then be used directly on the lot calculation, leading to larger long-term gains.


### This Repository

The db folder contains a module that strictly concerns itself with the scraping, storing, and visualization of raw data. Since the nature of volatile assets presents tiny opportunities to profitably buy and sell at any moment regardless of wider trend, I opted to walk away from querying trading platform APIs for minutely OHLV data, favoring the storage sub-minute instantaneous tick and technical indicator (RSI, MACD, EMA in various lengths,) exactly as they would appear in a live scenario, stored. The interval is adjustable, and the default is 5-second intervals.

The RL folder houses the logic that will produce our model. In a bid to decouple this approach from my own trading biases and observations, I opted to start the exploration off with a purely stochastic run. Just collect a large amount of ticks and indicators and let the simplest feed-forward network buy and sell at random while keeping track of reward and PnL states. From observing the output vector change in real time as the network works through the data, it is visually evident that converging values 