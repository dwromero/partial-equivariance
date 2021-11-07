#!/bin/bash
# Job requirements:
#SBATCH -N 1
# #SBATCH -t 2-00:00:00
#SBATCH -p clusterRTX
#SBATCH -x node03,node04
#SBATCH --gres=gpu:1
#SBATCH -n 4

# Load modules
module load anaconda3/5.1.0

# Activate environment
source activate partial_equiv_cu10.2

#
# nvidia-smi

# Run python file
export PYTHONPATH=$HOME/partial_equiv/partial_equivariance

cd $HOME/partial_equiv/partial_equivariance

python main.py base_group.name=E2 base_group.no_samples=16 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.partial_equiv=True dataset=STL10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[0,1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0001 train.lr_probs=1e-4

