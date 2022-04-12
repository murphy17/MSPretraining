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

# Notes from Apr 11 call with Kevin

- use early stopping
- The CARP model's final layer is logits, fix this
- Signal peptides: SignalP. Just pull random N-termini as negatives, or assign it to the right type of signal
- Use learned attention layer, 1 deep, to pool
- Use single-layer ReLU, no norm as final classifier
- Use the MS model as a convolutional filter, and run on one-vs-rest AAV/GB1 tasks (two-vs-rest for latter too); these are hard
- Baselines to worry about are (1) same arch, no pretraining; (2) really simple thing, like CNN or log-reg