# TODO

- Make sure the density->molarity calculation was correct (enough)

- Train-test split DBAASP
    - See how the Capecchi paper did the split

- Proximity in sequence distance between DBAASP sequences and ProteomeTools

- Fit single-task transformer to most represented species in DBAASP, initializing from random. Do it as a regression problem. CV search to find best architecture
    - How do I go from sequence-length encodings to scalar output? -> Just generate prediction at each position (exact same topology as MS) and then average-pool
    - https://botorch.org/

- Pretrain that exact architecture on the MS data
    - Why would this be appropriate? Doesn't it call for completely different decoder?