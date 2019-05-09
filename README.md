# randomNAS_release
Code release for paper "Random Search and Reproducibility for NAS."

This code base requires the following additional repositories:  
<https://github.com/liamcli/darts>  
<https://github.com/liamcli/darts_asha>

Random seeds and additional documentation needed to reproduce the results are provided in [this spreadsheet](https://docs.google.com/spreadsheets/d/1XajrgOnNr7rST8sDYX8YVV_IHYlI98h21JRph0Uz6QU/edit?usp=sharing).

You will need the following packages to run the code:  
`Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0, CUDA 9.0, CuDNN 7.1.3`

Download the CIFAR-10 dataset and PTB dataset according to how DARTS expects the files to be.

## Random Search with Weight Sharing
Be sure to set the right paths for the `darts` repository and data files in the benchmark files:  
`benchmarks/cnn/darts/darts_wrapper_discrete.py`  
`benchmarks/ptb/darts/darts_wrapper_discrete.py`

To run `random_weight_share.py`, issue the command with the desired arguments to replicate runs from the spreadsheet:  
`python random_weight_share.py --benchmark [ptb/cnn] --seed [X] --epochs [X] --batch_size [X] --grad_clip [X] --save_dir [X] --config [controls ptb hidden dim] --init_channels [controls cnn size]`

Please use our fork of DARTS in order to get deterministic results for CNN evaluation.  The output architecture for PTB can be copied directly into `darts/rnn/genotypes.py` for eval.  For CNN, run the following command to get the architecture to copy into `darts/cnn/genotypes.py` for eval:  
`python parse_cnn_arch.py "[arch_str]"`

Note that the exact reproducibility of random search with weight sharing is **hardware dependent**; our CNN results are reproducible on Tesla V100 GPUs and our RNN results are reproducible on Tesla P100 GPUs.  

## ASHA
Please follow the directions in the `darts_asha` repo to run these experiments.
