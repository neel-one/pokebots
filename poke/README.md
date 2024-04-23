Old approach to creating Pokemon Showdown bots.
This approach played with using Reinforcement Learning using mushroom-rl
This approach also tries to use the data parsed from poke-env to vectorize the data. 

This approach overall is too complicated and bound to fail:
1. RL with self-play is too bug prone. For example, there are random self-play bugs that are hard to diagnose, is unstable and can freeze the training run and it is hard to error-handle/debug. There's too much of a reliance on the Pokemon Showdown server
2. RL is slow
3. To train the model, we do try to build state vectors of each move using the poke-env framework (not sure why its in greedy.py). This is promising but poses a few issues: there's just not enough data to be grabbed automatically from the local run of Pokemon Showdown. Consequently, the next followup is to scrape game logs from Pokemon Showdown server and pass it through poke-env to build the same state model. This is unfortunately is quite hard because you have to make some non-trivial changes in the poke-env framework itself, and its a bit hacky because the framework is built to consume user logs from the Pokemon Showdown protocol not lines of the replay logs (which slightly differ from user real-time protocol logs).

Due to these problems, we need a better approach. As such, I've begun rearchitecting this in the `poke-v2` directory. We eliminate our reliance on the poke-env framework. Instead, we directly scrape raw game logs from Pokemon Showdown replays, build our dataset from the raw replays, and train a model on that dataset. 