The guide is desgined for an AI planning agent to create detailed instructions for a AI coding agent to create an options strategy backtester in python. 

To maximize context window, first create instructions for an overall program and file structure; we intend to create each structure in separate runs in order to maximize context. Any notable details and important summaries of the context should be stored and referenced from the contextSummary folder. 

After creating each component of the code, make sure to check it runs bug free and is coherent with the rest of the code base, making full use of the code within the repo. 

# Code Structure Guidelines 

The following structure and objects are needed:

## Option
The Option class object is the minimal building block of an option structure. It can only be one of the 4 options: long call, short call, long put, and short put. It includes all the information regarding that single option trade, including strike, maturity, day 1 cost(premium credit/debit), size, direction, greeks etc.

## OptionStructure
The OptionStructure class object is a collection of Option objects. It contains aggregate information of its Option objects and allows easier entering and exiting of options positions in Strategies. For example, a LongStraddle OptionStructure object(that is long the same amount of call and put of the same strike and maturity) created by inheriting OptionsStructure, would only require the number of contracts, size, and maturity to quickly implement the strategy.

## Strategy
The Strategy class object is a collection of OptionsStructure objects and other Strategy class objects with detailed conditions on when to enter and exit them. It is able to stream timeseries data during back testing through a backtesting engine and through if/else conditions decede when to enter and exit option positions. 

## Backtesting Engine
The backtesting engine class is able to load and stream time series data through multiple Strategy class objects and backtest the performance of these strategies collecting different metrics. The backtester must be able to present visualization and summary.

# Agents to be created

Created one quantiative developer agent that writes code. Create another agent that checks and tests the code such that the code is bug free and runs coherently with rest of the codebase. 

# Overall guideline. 
Use uv to create virtual enivornments and install any dependencies. Reference Install.md as needed.
Make sure Dolt is implemented as a way to access data. Reference Dolt.md thoroughly. "dolt clone post-no-preference/options" we wish to use this database, also accesible through this link: https://www.dolthub.com/repositories/post-no-preference/options/doc/master

Store the instructions as CODING_INSTRUCTION.md within the instructions folder.

