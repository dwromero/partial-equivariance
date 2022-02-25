#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=24:00:00
ls
pwd

module load anaconda3/personal

nvidia-smi

echo "starting script."

python $HOME/partial-equivariance/main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=300 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0 train.lr_probs=1e-4 conv.mask=False
python $HOME/partial-equivariance/main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=301 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0 train.lr_probs=1e-4 conv.mask=False
python $HOME/partial-equivariance/main.py base_group.name=SE2 base_group.no_samples=1 base_group.sample_per_batch_element=False base_group.sample_per_layer=False base_group.sampling_method=deterministic conv.bias=True conv.padding=same conv.partial_equiv=False dataset=CIFAR10 kernel.learn_omega0=False kernel.no_hidden=32 kernel.no_layers=3 kernel.omega0=10 kernel.size=7 kernel.type=SIREN kernel.weight_norm=False net.dropout=0 net.no_blocks=2 net.no_hidden=32 net.norm=BatchNorm net.pool_blocks=[1,2] net.block_width_factors=[1,1,2,1] net.type=CKResNet no_workers=3 seed=302 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001 train.scheduler=cosine train.scheduler_params.warmup_epochs=5 train.weight_decay=0.0 train.lr_probs=1e-4 conv.mask=False


echo "script done."