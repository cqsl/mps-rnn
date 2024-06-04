#!/bin/sh

HAM_ARGS='--ham heis_tri --J afm --boundary open --sign mars --ham_dim 2 --L 2'
NET_ARGS_MPS='--bond_dim 2 --zero_mag --reorder_type snake --dtype complex128 --seed 123'
NET_ARGS="$NET_ARGS_MPS --affine"

./test_autoreg.py $HAM_ARGS --net mps --net_dim 1 $NET_ARGS_MPS || exit
./test_autoreg.py $HAM_ARGS --net mps_rnn --net_dim 1 $NET_ARGS || exit
./test_autoreg.py $HAM_ARGS --net mps_rnn --net_dim 2 $NET_ARGS || exit
./test_autoreg.py $HAM_ARGS --net tensor_rnn_cmpr --net_dim 2 $NET_ARGS || exit
./test_autoreg.py $HAM_ARGS --net tensor_rnn --net_dim 2 $NET_ARGS || exit

./test_autoreg_cond.py $HAM_ARGS --net mps --net_dim 1 $NET_ARGS_MPS || exit
./test_autoreg_cond.py $HAM_ARGS --net mps_rnn --net_dim 1 $NET_ARGS || exit
./test_autoreg_cond.py $HAM_ARGS --net mps_rnn --net_dim 2 $NET_ARGS || exit
./test_autoreg_cond.py $HAM_ARGS --net tensor_rnn_cmpr --net_dim 2 $NET_ARGS || exit
./test_autoreg_cond.py $HAM_ARGS --net tensor_rnn --net_dim 2 $NET_ARGS || exit

./test_autoreg_sample.py $HAM_ARGS --net mps --net_dim 1 $NET_ARGS_MPS || exit
./test_autoreg_sample.py $HAM_ARGS --net mps_rnn --net_dim 1 $NET_ARGS || exit
./test_autoreg_sample.py $HAM_ARGS --net mps_rnn --net_dim 2 $NET_ARGS || exit
./test_autoreg_sample.py $HAM_ARGS --net tensor_rnn_cmpr --net_dim 2 $NET_ARGS || exit
./test_autoreg_sample.py $HAM_ARGS --net tensor_rnn --net_dim 2 $NET_ARGS || exit

echo 'All tests passed'
