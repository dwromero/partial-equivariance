#!/bin/bash
cd ..
python main.py base_group.name=SE2 base_group.no_samples=8 base_group.sample_per_batch_element=False base_group.sample_per_layer=True base_group.sampling_method=random conv.bias=True conv.padding=same conv.cond_rot=True dataset=MNIST6-180 kernel.no_hidden=32 kernel.no_layers=3 kernel.size=7 kernel.type=SIREN kernel.learn_omega0=False kernel.omega0=10.0 kernel.learn_omega1=False kernel.omega1=10.0 net.dropout=0 net.no_blocks=2 net.no_hidden=10 net.norm=BatchNorm net.pool_blocks=[1,2] net.type=CKResNet seed=0 train.batch_size=64 train.do=True train.epochs=300 train.lr=0.001
