# TODO

- Rerun the MS pretrainer. I'm worried the train-test split was contaminated

- Fit single-task transformer to most represented species in DBAASP, initializing from random. Do it as a regression problem. CV search to find best architecture
    - How do I go from sequence-length encodings to scalar output? -> Just generate prediction at each position (exact same topology as MS) and then average-pool
    - only use the encoder, then average-pool and pass into a dense model
    - https://botorch.org/

- Pretrain that exact architecture on the MS data
    - Why would this be appropriate? Doesn't it call for completely different decoder?
    - maybe not. just ditch the positional encoding on the targets, and average-pool at the end
    
    
    
- Make sure the density->molarity calculation was correct (or just Z-score)

- Proximity in sequence distance between DBAASP sequences and ProteomeTools
