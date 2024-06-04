#!/bin/sh

# You may add `--refl_sym` but this will significantly increase training time
HAM_DIR=heis_tri_afm_mars_2d_L10
NET_DIR=chi16_zm_af_snake_sc_gc1
HAM_ARGS='--ham heis_tri --J afm --boundary open --sign mars --ham_dim 2 --L 10'
NET_ARGS='--bond_dim 16 --zero_mag --affine --reorder_type snake --dtype complex64'
OPT_ARGS='--seed 123 --optimizer adam --split_complex --batch_size 1024 --lr 0.001 --max_step 10000 --grad_clip 1'

# DMRG
mkdir out
cd dmrg
julia -e 'using Pkg; Pkg.activate("."); include("./heisenberg_2d.jl"); main(L=10, max_B=16, J2=1, seed=123);' > ../out/dmrg.log
cd ..

# 1D MPS-RNN
mkdir -p out/$HAM_DIR/mps_rnn_1d_$NET_DIR
mv dmrg/L10_open_mars_J2\=1_B16_zm.hdf5 out/$HAM_DIR/mps_rnn_1d_$NET_DIR/init.hdf5
# You may add `--chunk_size 1024` if out of memory,
# and `--show_progress` if running interactively
./vmc.py $HAM_ARGS --net mps_rnn --net_dim 1 $NET_ARGS $OPT_ARGS

# 2D MPS-RNN
mkdir -p out/$HAM_DIR/mps_rnn_2d_$NET_DIR
cp out/$HAM_DIR/mps_rnn_1d_$NET_DIR/out.mpack out/$HAM_DIR/mps_rnn_2d_$NET_DIR/init_hi.mpack
./vmc.py $HAM_ARGS --net mps_rnn --net_dim 2 $NET_ARGS $OPT_ARGS

# Compressed tensor-RNN
mkdir -p out/$HAM_DIR/tensor_rnn_cmpr_2d_$NET_DIR
cp out/$HAM_DIR/mps_rnn_2d_$NET_DIR/out.mpack out/$HAM_DIR/tensor_rnn_cmpr_2d_$NET_DIR/init_hi.mpack
./vmc.py $HAM_ARGS --net tensor_rnn_cmpr --net_dim 2 $NET_ARGS $OPT_ARGS

# Tensor-RNN
mkdir -p out/$HAM_DIR/tensor_rnn_2d_$NET_DIR
cp out/$HAM_DIR/tensor_rnn_cmpr_2d_$NET_DIR/out.mpack out/$HAM_DIR/tensor_rnn_2d_$NET_DIR/init_hi.mpack
./vmc.py $HAM_ARGS --net tensor_rnn --net_dim 2 $NET_ARGS $OPT_ARGS
