# What is found after steps/nnet3/chain/train.py

We find log files in a timely mannter and see what commands are executed in order. First, list up log files.

```
ls exp/multi_a/tdnn_7k_rvb1/log/* -haltr | head
```

You many need to check more precise time information using stat command.

## get_lda_stats.1.log

```
stat exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.1.log
  File: `exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.1.log'
  Size: 1058            Blocks: 8          IO Block: 4096   일반 파일
Device: 811h/2065d      Inode: 77729266    Links: 1
Access: (0644/-rw-r--r--)  Uid: (    0/    root)   Gid: (    0/    root)
Access: 2018-05-03 08:02:17.490870600 +0900
Modify: 2018-05-02 03:43:58.870485847 +0900
Change: 2018-05-02 03:43:58.870485847 +0900
 Birth: -

cat exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.1.log
# nnet3-chain-acc-lda-stats --rand-prune=4.0 exp/multi_a/tdnn_7k_rvb1/init.raw ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.1.ark exp/multi_a/tdnn_7k_rvb1/1.lda_stats 
# Started at Wed May  2 03:43:35 KST 2018
#
nnet3-chain-acc-lda-stats --rand-prune=4.0 exp/multi_a/tdnn_7k_rvb1/init.raw ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.1.ark exp/multi_a/tdnn_7k_rvb1/1.lda_stats 
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:main():nnet3-chain-acc-lda-stats.cc:195) Processed 10150 examples.
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:WriteStats():nnet3-chain-acc-lda-stats.cc:67) Accumulated stats, soft frame count = 506092.  Wrote to exp/multi_a/tdnn_7k_rvb1/1.lda_stats
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.00966 seconds taken in nnet3 compilation total (breakdown: 0.000514 compilation, 6.41e-05 optimization, 0 shortcut expansion, 4.22e-05 checking, 6.03e-06 computing indexes, 0.00903 misc.)
# Accounting: time=23 threads=1
# Ended (code 0) at Wed May  2 03:43:58 KST 2018, elapsed time 23 seconds
```

## init_mdl.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/init_mdl.log
# nnet3-am-init exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/0.raw exp/multi_a/tdnn_7k_rvb1/0.mdl 
# Started at Wed May  2 03:44:02 KST 2018
#
nnet3-am-init exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/0.raw exp/multi_a/tdnn_7k_rvb1/0.mdl 
LOG (nnet3-am-init[5.2.107~2-e892]:main():nnet3-am-init.cc:96) Initialized am-nnet (neural net acoustic model) and wrote to exp/multi_a/tdnn_7k_rvb1/0.mdl
# Accounting: time=3 threads=1
# Ended (code 0) at Wed May  2 03:44:05 KST 2018, elapsed time 3 seconds
root@MM-8GPU02:/home/kwon/s1# cat exp/multi_a/tdnn_7k_rvb1/log/init_mdl.log
# nnet3-am-init exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/0.raw exp/multi_a/tdnn_7k_rvb1/0.mdl 
# Started at Wed May  2 03:44:02 KST 2018
#
nnet3-am-init exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/0.raw exp/multi_a/tdnn_7k_rvb1/0.mdl 
LOG (nnet3-am-init[5.2.107~2-e892]:main():nnet3-am-init.cc:96) Initialized am-nnet (neural net acoustic model) and wrote to exp/multi_a/tdnn_7k_rvb1/0.mdl
# Accounting: time=3 threads=1
# Ended (code 0) at Wed May  2 03:44:05 KST 2018, elapsed time 3 seconds
```

## compute_prob_train.0.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.0.log
# nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 "nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/train_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" 
# Started at Wed May  2 03:44:05 KST 2018
#
nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 'nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/train_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |' 
nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/0.mdl to raw format as -
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/train_diagnostic.cegs ark:- 
nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- 
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 400 neural-network training examples, wrote 400
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={16->1,64->6,d=0}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 400 egs of avg. size 181 into 7 minibatches, discarding 0% of egs.  Avg minibatch size was 57.14, #distinct types of egs/minibatches was 1/2
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:193) Overall log-probability for 'output-xent' is -9.31781 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:193) Overall log-probability for 'output' is -1.07591 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.0214 seconds taken in nnet3 compilation total (breakdown: 0.017 compilation, 0.00132 optimization, 0.000995 shortcut expansion, 0.000333 checking, 9.38e-05 computing indexes, 0.00161 misc.)
# Accounting: time=87 threads=1
# Ended (code 0) at Wed May  2 03:45:32 KST 2018, elapsed time 87 seconds
```

## compute_prob_valid.0.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.0.log
# nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 "nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" 
# Started at Wed May  2 03:44:05 KST 2018
#
nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 'nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |' 
nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/0.mdl to raw format as -
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
WARNING (nnet3-chain-compute-prob[5.2.107~2-e892]:ComputeDerived():nnet-simple-component.cc:5296) Test-mode is set but there is no data count.  Creating random counts.  This only makes sense in unit-tests (or compute_prob_*.0.log).  If you see this elsewhere, something is very wrong.
nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs ark:- 
nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- 
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 400 neural-network training examples, wrote 400
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={16->1,64->6,d=0}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 400 egs of avg. size 181 into 7 minibatches, discarding 0% of egs.  Avg minibatch size was 57.14, #distinct types of egs/minibatches was 1/2
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:193) Overall log-probability for 'output-xent' is -9.31781 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:193) Overall log-probability for 'output' is -1.08398 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.0175 seconds taken in nnet3 compilation total (breakdown: 0.0128 compilation, 0.00133 optimization, 0.00122 shortcut expansion, 0.00032 checking, 0.000156 computing indexes, 0.00168 misc.)
# Accounting: time=87 threads=1
# Ended (code 0) at Wed May  2 03:45:32 KST 2018, elapsed time 87 seconds
```

## select.0.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/select.0.log
# nnet3-copy exp/multi_a/tdnn_7k_rvb1/1.3.raw - | nnet3-am-copy --set-raw-nnet=- exp/multi_a/tdnn_7k_rvb1/0.mdl exp/multi_a/tdnn_7k_rvb1/1.mdl 
# Started at Wed May  2 03:45:44 KST 2018
#
nnet3-am-copy --set-raw-nnet=- exp/multi_a/tdnn_7k_rvb1/0.mdl exp/multi_a/tdnn_7k_rvb1/1.mdl 
nnet3-copy exp/multi_a/tdnn_7k_rvb1/1.3.raw - 
LOG (nnet3-copy[5.2.107~2-e892]:main():nnet3-copy.cc:114) Copied raw neural net from exp/multi_a/tdnn_7k_rvb1/1.3.raw to -
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:147) Copied neural net from exp/multi_a/tdnn_7k_rvb1/0.mdl to exp/multi_a/tdnn_7k_rvb1/1.mdl
# Accounting: time=1 threads=1
# Ended (code 0) at Wed May  2 03:45:45 KST 2018, elapsed time 1 seconds
```

## progress.1.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/progress.1.log
# nnet3-am-info exp/multi_a/tdnn_7k_rvb1/1.mdl && nnet3-show-progress --use-gpu=no "nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |" "nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - |" 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-am-info exp/multi_a/tdnn_7k_rvb1/1.mdl 
input-dim: 40
ivector-dim: 100
num-pdfs: 11130
prior-dimension: 0
# Nnet info follows.
left-context: 17
right-context: 12
num-parameters: 41924340
modulus: 1
input-node name=ivector dim=100
input-node name=input dim=40
component-node name=lda component=lda input=Append(Offset(input, -1), input, Offset(input, 1), ReplaceIndex(ivector, t, 0)) input-dim=220 output-dim=220
component-node name=tdnn1.affine component=tdnn1.affine input=lda input-dim=220 output-dim=1024
component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine input-dim=1024 output-dim=1024
component-node name=tdnn1.batchnorm component=tdnn1.batchnorm input=tdnn1.relu input-dim=1024 output-dim=1024
component-node name=tdnn2.affine component=tdnn2.affine input=Append(Offset(tdnn1.batchnorm, -1), tdnn1.batchnorm, Offset(tdnn1.batchnorm, 1), Offset(tdnn1.batchnorm, 2)) input-dim=4096 output-dim=1024
component-node name=tdnn2.relu component=tdnn2.relu input=tdnn2.affine input-dim=1024 output-dim=1024
component-node name=tdnn2.batchnorm component=tdnn2.batchnorm input=tdnn2.relu input-dim=1024 output-dim=1024
component-node name=tdnn3.affine component=tdnn3.affine input=Append(Offset(tdnn2.batchnorm, -3), tdnn2.batchnorm, Offset(tdnn2.batchnorm, 3)) input-dim=3072 output-dim=1024
component-node name=tdnn3.relu component=tdnn3.relu input=tdnn3.affine input-dim=1024 output-dim=1024
component-node name=tdnn3.batchnorm component=tdnn3.batchnorm input=tdnn3.relu input-dim=1024 output-dim=1024
component-node name=tdnn4.affine component=tdnn4.affine input=Append(Offset(tdnn3.batchnorm, -3), tdnn3.batchnorm, Offset(tdnn3.batchnorm, 3)) input-dim=3072 output-dim=1024
component-node name=tdnn4.relu component=tdnn4.relu input=tdnn4.affine input-dim=1024 output-dim=1024
component-node name=tdnn4.batchnorm component=tdnn4.batchnorm input=tdnn4.relu input-dim=1024 output-dim=1024
component-node name=tdnn5.affine component=tdnn5.affine input=Append(Offset(tdnn4.batchnorm, -3), tdnn4.batchnorm, Offset(tdnn4.batchnorm, 3)) input-dim=3072 output-dim=1024
component-node name=tdnn5.relu component=tdnn5.relu input=tdnn5.affine input-dim=1024 output-dim=1024
component-node name=tdnn5.batchnorm component=tdnn5.batchnorm input=tdnn5.relu input-dim=1024 output-dim=1024
component-node name=tdnn6.affine component=tdnn6.affine input=Append(Offset(tdnn5.batchnorm, -6), Offset(tdnn5.batchnorm, -3), tdnn5.batchnorm) input-dim=3072 output-dim=1024
component-node name=tdnn6.relu component=tdnn6.relu input=tdnn6.affine input-dim=1024 output-dim=1024
component-node name=tdnn6.batchnorm component=tdnn6.batchnorm input=tdnn6.relu input-dim=1024 output-dim=1024
component-node name=prefinal-chain.affine component=prefinal-chain.affine input=tdnn6.batchnorm input-dim=1024 output-dim=1024
component-node name=prefinal-chain.relu component=prefinal-chain.relu input=prefinal-chain.affine input-dim=1024 output-dim=1024
component-node name=prefinal-chain.batchnorm component=prefinal-chain.batchnorm input=prefinal-chain.relu input-dim=1024 output-dim=1024
component-node name=output.affine component=output.affine input=prefinal-chain.batchnorm input-dim=1024 output-dim=11130
output-node name=output input=output.affine dim=11130 objective=linear
component-node name=prefinal-xent.affine component=prefinal-xent.affine input=tdnn6.batchnorm input-dim=1024 output-dim=1024
component-node name=prefinal-xent.relu component=prefinal-xent.relu input=prefinal-xent.affine input-dim=1024 output-dim=1024
component-node name=prefinal-xent.batchnorm component=prefinal-xent.batchnorm input=prefinal-xent.relu input-dim=1024 output-dim=1024
component-node name=output-xent.affine component=output-xent.affine input=prefinal-xent.batchnorm input-dim=1024 output-dim=11130
component-node name=output-xent.log-softmax component=output-xent.log-softmax input=output-xent.affine input-dim=11130 output-dim=11130
output-node name=output-xent input=output-xent.log-softmax dim=11130 objective=linear
component name=lda type=FixedAffineComponent, input-dim=220, output-dim=220, linear-params-rms=0.01028, bias-{mean,stddev}=0.0153,0.3634
component name=tdnn1.affine type=NaturalGradientAffineComponent, input-dim=220, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.06619, bias-{mean,stddev}=-0.02067,0.9811, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn1.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=4.98e+05, self-repaired-proportion=0.39034, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0,0,0,0 0,3e-09,0.06,0.83,1.2 1.6,2.0,2.2,3.5), mean=0.393, stddev=0.571], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0,0,0,0 0,1e-06,0.43,1.0,1.0 1.0,1.0,1.0,1.0), mean=0.487, stddev=0.46]
component name=tdnn1.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=893599, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0,0,0,0 0,3e-08,0.06,0.83,1.2 1.6,2.0,2.2,3.4), mean=0.393, stddev=0.571], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0,0,0,0 0,3e-05,0.10,0.17,0.19 0.20,0.23,0.24,0.31), mean=0.0894, stddev=0.0815]
component name=tdnn2.affine type=NaturalGradientAffineComponent, input-dim=4096, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.01566, bias-{mean,stddev}=0.04038,1.018, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn2.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.32e+05, self-repaired-proportion=0.0991309, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(1e-06,0.0006,0.004,0.01 0.03,0.07,0.34,0.99,1.4 1.7,2.1,2.3,3.5), mean=0.542, stddev=0.562], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(7e-06,0.001,0.009,0.03 0.06,0.15,0.52,0.89,0.95 0.98,0.99,1.0,1.0), mean=0.513, stddev=0.328]
component name=tdnn2.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=296164, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(7e-07,0.0004,0.004,0.01 0.03,0.07,0.34,0.99,1.4 1.7,2.1,2.3,3.5), mean=0.543, stddev=0.562], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.0004,0.02,0.06,0.11 0.17,0.25,0.50,0.71,0.78 0.84,0.90,0.94,1.1), mean=0.488, stddev=0.231]
component name=tdnn3.affine type=NaturalGradientAffineComponent, input-dim=3072, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.01802, bias-{mean,stddev}=-0.01053,0.9558, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn3.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.27e+05, self-repaired-proportion=0.0388506, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.006,0.01,0.02,0.04 0.07,0.14,0.42,0.93,1.3 1.6,1.9,2.2,3.4), mean=0.563, stddev=0.509], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.008,0.02,0.02,0.06 0.10,0.20,0.49,0.78,0.88 0.94,0.96,0.98,1.0), mean=0.491, stddev=0.28]
component name=tdnn3.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=285952, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.006,0.01,0.02,0.04 0.07,0.14,0.42,0.92,1.3 1.6,1.9,2.2,3.4), mean=0.562, stddev=0.509], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.10,0.17,0.20,0.27 0.33,0.43,0.65,0.87,0.96 1.0,1.1,1.1,1.2), mean=0.648, stddev=0.234]
component name=tdnn4.affine type=NaturalGradientAffineComponent, input-dim=3072, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.01796, bias-{mean,stddev}=-0.004728,0.9668, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn4.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.41e+05, self-repaired-proportion=0.0341228, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.008,0.02,0.02,0.04 0.07,0.13,0.42,0.95,1.3 1.6,1.9,2.2,3.5), mean=0.572, stddev=0.511], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.005,0.02,0.02,0.05 0.09,0.18,0.50,0.79,0.89 0.93,0.96,0.98,1.0), mean=0.492, stddev=0.29]
component name=tdnn4.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=275739, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.008,0.02,0.02,0.05 0.07,0.13,0.43,0.95,1.3 1.6,1.9,2.2,3.5), mean=0.573, stddev=0.511], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.16,0.23,0.25,0.30 0.38,0.48,0.69,0.90,0.98 1.0,1.1,1.1,1.2), mean=0.682, stddev=0.225]
component name=tdnn5.affine type=NaturalGradientAffineComponent, input-dim=3072, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.01791, bias-{mean,stddev}=0.00572,0.9749, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn5.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.23e+05, self-repaired-proportion=0.0490323, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.006,0.02,0.03,0.05 0.08,0.14,0.42,0.99,1.3 1.6,2.0,2.2,3.6), mean=0.582, stddev=0.517], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.007,0.01,0.02,0.05 0.09,0.19,0.49,0.80,0.89 0.93,0.96,0.98,0.99), mean=0.493, stddev=0.291]
component name=tdnn5.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=265527, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.006,0.02,0.03,0.05 0.08,0.15,0.42,0.99,1.3 1.6,2.0,2.2,3.6), mean=0.583, stddev=0.517], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.12,0.30,0.32,0.36 0.43,0.52,0.71,0.91,0.99 1.1,1.1,1.2,1.3), mean=0.712, stddev=0.211]
component name=tdnn6.affine type=NaturalGradientAffineComponent, input-dim=3072, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.01782, bias-{mean,stddev}=0.01957,0.9758, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=tdnn6.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.16e+05, self-repaired-proportion=0.0533485, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.01,0.02,0.03,0.05 0.09,0.15,0.43,0.99,1.3 1.7,2.1,2.3,2.8), mean=0.597, stddev=0.529], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.003,0.01,0.02,0.05 0.10,0.20,0.47,0.79,0.88 0.93,0.97,0.98,1.0), mean=0.489, stddev=0.286]
component name=tdnn6.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=1, count=255314, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.01,0.02,0.03,0.05 0.09,0.15,0.42,0.99,1.3 1.7,2.1,2.3,2.8), mean=0.596, stddev=0.529], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.15,0.26,0.30,0.38 0.44,0.52,0.72,0.93,1.0 1.1,1.2,1.2,1.4), mean=0.724, stddev=0.222]
component name=prefinal-chain.affine type=NaturalGradientAffineComponent, input-dim=1024, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.03076, bias-{mean,stddev}=0.005572,0.9954, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=prefinal-chain.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.36e+05, self-repaired-proportion=0.0519833, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.003,0.01,0.02,0.04 0.08,0.15,0.42,0.98,1.3 1.7,2.2,2.3,3.3), mean=0.592, stddev=0.545], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.002,0.01,0.02,0.06 0.10,0.19,0.47,0.79,0.88 0.93,0.97,0.98,1.0), mean=0.487, stddev=0.287]
component name=prefinal-chain.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=0.5, count=255314, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.002,0.01,0.02,0.04 0.07,0.15,0.42,0.98,1.3 1.7,2.2,2.3,3.3), mean=0.592, stddev=0.546], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.06,0.19,0.23,0.29 0.36,0.45,0.69,0.91,0.98 1.1,1.1,1.2,1.5), mean=0.684, stddev=0.24]
component name=output.affine type=NaturalGradientAffineComponent, input-dim=1024, output-dim=11130, learning-rate=0.004, max-change=1.5, linear-params-rms=0.003238, bias-{mean,stddev}=6.728e-05,0.01904, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=prefinal-xent.affine type=NaturalGradientAffineComponent, input-dim=1024, output-dim=1024, learning-rate=0.004, max-change=0.75, linear-params-rms=0.03024, bias-{mean,stddev}=-0.02636,1.003, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=prefinal-xent.relu type=RectifiedLinearComponent, dim=1024, self-repair-scale=1e-05, count=1.2e+05, self-repaired-proportion=0.0780283, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.002,0.01,0.02,0.03 0.05,0.10,0.34,0.91,1.3 1.7,2.2,2.5,3.7), mean=0.541, stddev=0.562], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.003,0.01,0.02,0.05 0.08,0.15,0.45,0.82,0.91 0.96,0.99,0.99,1.0), mean=0.481, stddev=0.308]
component name=prefinal-xent.batchnorm type=BatchNormComponent, dim=1024, block-dim=1024, epsilon=0.001, target-rms=0.5, count=255314, test-mode=false, data-mean=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.002,0.01,0.02,0.03 0.05,0.10,0.34,0.91,1.3 1.7,2.2,2.5,3.7), mean=0.542, stddev=0.562], data-stddev=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.06,0.14,0.17,0.21 0.26,0.37,0.59,0.81,0.89 0.96,1.0,1.1,1.3), mean=0.591, stddev=0.236]
component name=output-xent.affine type=NaturalGradientAffineComponent, input-dim=1024, output-dim=11130, learning-rate=0.02, learning-rate-factor=5, max-change=1.5, linear-params-rms=0.005102, bias-{mean,stddev}=4.832e-08,0.04713, rank-in=20, rank-out=80, num-samples-history=2000, update-period=4, alpha=4
component name=output-xent.log-softmax type=LogSoftmaxComponent, dim=11130
nnet3-show-progress --use-gpu=no 'nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - |' 'nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - |' 
LOG (nnet3-show-progress[5.2.107~2-e892]:SelectGpuId():cu-device.cc:110) Manually selected to compute on CPU.
nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/0.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/0.mdl to raw format as -
nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
LOG (nnet3-show-progress[5.2.107~2-e892]:main():nnet3-show-progress.cc:139) Parameter differences per layer are [ tdnn1.affine:4.99759 tdnn2.affine:4.58228 tdnn3.affine:4.77428 tdnn4.affine:4.83238 tdnn5.affine:4.8644 tdnn6.affine:5.03429 prefinal-chain.affine:4.87533 output.affine:11.1146 prefinal-xent.affine:2.9652 output-xent.affine:17.9281 ]
LOG (nnet3-show-progress[5.2.107~2-e892]:main():nnet3-show-progress.cc:146) Norms of parameter matrices are [ tdnn1.affine:44.9323 tdnn2.affine:45.7235 tdnn3.affine:44.2977 tdnn4.affine:44.5314 tdnn5.affine:44.768 tdnn6.affine:44.8202 prefinal-chain.affine:45.3269 output.affine:0 prefinal-xent.affine:45.5076 output-xent.affine:0 ]
LOG (nnet3-show-progress[5.2.107~2-e892]:main():nnet3-show-progress.cc:150) Relative parameter differences per layer are [ tdnn1.affine:0.111225 tdnn2.affine:0.100217 tdnn3.affine:0.107777 tdnn4.affine:0.108516 tdnn5.affine:0.108658 tdnn6.affine:0.112322 prefinal-chain.affine:0.107559 output.affine:-nan prefinal-xent.affine:0.0651584 output-xent.affine:inf ]
# Accounting: time=1 threads=1
# Ended (code 0) at Wed May  2 03:45:46 KST 2018, elapsed time 1 seconds
```

## compute_prob_valid.1.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.1.log
# nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 "nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-chain-compute-prob --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 'nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |' 
nnet3-am-copy --raw=true exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
nnet3-chain-copy-egs ark:exp/multi_a/tdnn_7k_rvb1/egs/valid_diagnostic.cegs ark:- 
nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- 
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 400 neural-network training examples, wrote 400
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={16->1,64->6,d=0}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 400 egs of avg. size 181 into 7 minibatches, discarding 0% of egs.  Avg minibatch size was 57.14, #distinct types of egs/minibatches was 1/2
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:193) Overall log-probability for 'output-xent' is -7.25934 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:PrintTotalStats():nnet-chain-diagnostics.cc:198) Overall log-probability for 'output' is -0.638957 + -0.0101632 = -0.64912 per frame, over 20000 frames.
LOG (nnet3-chain-compute-prob[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.0227 seconds taken in nnet3 compilation total (breakdown: 0.0179 compilation, 0.00144 optimization, 0.00106 shortcut expansion, 0.000352 checking, 0.000102 computing indexes, 0.00179 misc.)
# Accounting: time=83 threads=1
# Ended (code 0) at Wed May  2 03:47:08 KST 2018, elapsed time 83 seconds
```
