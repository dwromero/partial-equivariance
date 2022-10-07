# Experiments

We provide the commands used to run the experiments published in the paper. For each experiment we provide a single command. For experiments where results are reported over multiple runs one should use incremental integer seeds starting at zero to reproduce the original results. For example, for an experiment with three runs we used `seed=0`, `seed=1` and `seed=2`.

Please note that due to randomness in certain PyTorch operations on CUDA, it may not be possible to reproduce certain results with high precision. Please see [PyTorch's manual on deterministic behavior](https://pytorch.org/docs/stable/notes/randomness.html) for more details, as well as `run_experiments.py::set_manual_seed()` for specifications on how we seed our experiments.