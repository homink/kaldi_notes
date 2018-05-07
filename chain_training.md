# What is found after steps/nnet3/chain/train.py

Let us have a look at the brife flow of [steps/nnet3/chain/train.py](https://github.com/kaldi-asr/kaldi/blob/e89280576107fcac7ad4d1b95eb8eaf8164bdccd/egs/wsj/s5/steps/nnet3/chain/train.py) [(e892805)](https://github.com/kaldi-asr/kaldi/tree/e89280576107fcac7ad4d1b95eb8eaf8164bdccd)

```
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.chain_objf.acoustic_model as chain_lib

def train(args, run_opts):

    chain_lib.check_for_required_files(args.feat_dir, args.tree_dir,
                                       args.lat_dir)

    common_lib.execute_command("utils/split_data.sh {0} {1}".format(
            args.feat_dir, num_jobs))

    variables = common_train_lib.parse_generic_config_vars_file(var_file)

    if (args.stage <= -6):
        chain_lib.create_phone_lm(args.dir, args.tree_dir, run_opts,
                                  lm_opts=args.lm_opts)

    if (args.stage <= -5):
        chain_lib.create_denominator_fst(args.dir, args.tree_dir, run_opts)

    if (args.stage <= -4) and os.path.exists(args.dir+"/configs/init.config"):
        common_lib.execute_command(
            """{command} {dir}/log/nnet_init.log \
                    nnet3-init --srand=-2 {dir}/configs/init.config \
                    {dir}/init.raw""".format(command=run_opts.command,
                                             dir=args.dir))

    if (args.stage <= -3) and args.egs_dir is None:
        chain_lib.generate_chain_egs(...)

    [egs_left_context, egs_right_context,
     frames_per_eg_str, num_archives] = (
        common_train_lib.verify_egs_dir(...))

    common_train_lib.copy_egs_properties_to_exp_dir(egs_dir, args.dir)

    if (args.stage <= -2) and os.path.exists(args.dir+"/configs/init.config"):
        chain_lib.compute_preconditioning_matrix(
            args.dir, egs_dir, num_archives, run_opts,
            max_lda_jobs=args.max_lda_jobs,
            rand_prune=args.rand_prune)

    if (args.stage <= -1):
        chain_lib.prepare_initial_acoustic_model(args.dir, run_opts)

    for iter in range(num_iters):
        if args.stage <= iter:
            chain_lib.train_one_iteration(...)

            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain
                # conditions
                common_train_lib.remove_model(...)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        if args.do_final_combination:
            chain_lib.combine_models(...)
        else:
            logger.info("Copying the last-numbered model to final.mdl")
            common_lib.force_symlink("{0}.mdl".format(num_iters),
                                     "{0}/final.mdl".format(args.dir))
            common_lib.force_symlink("compute_prob_valid.{iter}.log".format(
                                         iter=num_iters-1),
                                     "{dir}/log/compute_prob_valid.final.log".format(
                                         dir=args.dir))

    if args.cleanup:

        common_train_lib.clean_nnet_dir(...)


    common_lib.execute_command("steps/info/nnet3_dir_info.pl "
                               "{0}".format(args.dir))
```

In the training, TDNN network and input arguments for [steps/nnet3/chain/train.py](https://github.com/kaldi-asr/kaldi/blob/e89280576107fcac7ad4d1b95eb8eaf8164bdccd/egs/wsj/s5/steps/nnet3/chain/train.py) [(e892805)](https://github.com/kaldi-asr/kaldi/tree/e89280576107fcac7ad4d1b95eb8eaf8164bdccd) are configured as follows:

```
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$hidden_dim
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
```
```
2018-05-02 00:49:53,625 [steps/nnet3/chain/train.py:259 - train - INFO ] Arguments for the experiment
{'alignment_subsampling_factor': 3,
 'apply_deriv_weights': False,
 'backstitch_training_interval': 1,
 'backstitch_training_scale': 0.0,
 'chunk_left_context': 0,
 'chunk_left_context_initial': -1,
 'chunk_right_context': 0,
 'chunk_right_context_final': -1,
 'chunk_width': '150',
 'cleanup': True,
 'cmvn_opts': '--norm-means=false --norm-vars=false',
 'combine_sum_to_one_penalty': 0.0,
 'command': 'run.pl',
 'compute_per_dim_accuracy': False,
 'deriv_truncate_margin': None,
 'dir': 'exp/multi_a/tdnn_7k_rvb1',
 'do_final_combination': True,
 'dropout_schedule': None,
 'egs_command': 'queue.pl',
 'egs_dir': None,
 'egs_opts': '--frames-overlap-per-eg 0',
 'egs_stage': -10,
 'email': None,
 'exit_stage': None,
 'feat_dir': 'data/librispeech_ami_fisher/train_rvb1_min1.55_hires',
 'final_effective_lrate': 0.0001,
 'frame_subsampling_factor': 3,
 'frames_per_iter': 1500000,
 'initial_effective_lrate': 0.001,
 'l2_regularize': 5e-05,
 'lat_dir': 'exp/multi_a/tri7b_rvb_min1.55_lats',
 'leaky_hmm_coefficient': 0.1,
 'left_deriv_truncate': None,
 'left_tolerance': 5,
 'lm_opts': '--num-extra-lm-states=2000',
 'max_lda_jobs': 10,
 'max_models_combine': 20,
 'max_param_change': 2.0,
 'momentum': 0.0,
 'num_chunk_per_minibatch': '128',
 'num_epochs': 4.0,
 'num_jobs_final': 8,
 'num_jobs_initial': 4,
 'online_ivector_dir': 'exp/multi_a/ivectors_train_min1.55',
 'preserve_model_interval': 100,
 'presoftmax_prior_scale_power': -0.25,
 'proportional_shrink': 0.0,
 'rand_prune': 4.0,
 'remove_egs': True,
 'reporting_interval': 0.1,
 'right_tolerance': 5,
 'samples_per_iter': 400000,
 'shrink_saturation_threshold': 0.4,
 'shrink_value': 1.0,
 'shuffle_buffer_size': 5000,
 'srand': 0,
 'stage': -10,
 'transform_dir': 'exp/multi_a/tri7b_rvb_min1.55_lats',
 'tree_dir': 'exp/multi_a/tree_tdnn_7k_rvb1',
 'use_gpu': True,
 'xent_regularize': 0.1}
```

We find log files in a timely manner and see what commands are executed in order. First, list up log files. We can see some pattern. Let us browse them each.

```
ls exp/multi_a/tdnn_7k_rvb1/log/* -tr | head -30
exp/multi_a/tdnn_7k_rvb1/log/make_phone_lm.log
exp/multi_a/tdnn_7k_rvb1/log/make_den_fst.log
exp/multi_a/tdnn_7k_rvb1/log/nnet_init.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.9.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.10.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.8.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.7.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.2.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.1.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.6.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.3.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.4.log
exp/multi_a/tdnn_7k_rvb1/log/get_lda_stats.5.log
exp/multi_a/tdnn_7k_rvb1/log/sum_transform_stats.log
exp/multi_a/tdnn_7k_rvb1/log/get_transform.log
exp/multi_a/tdnn_7k_rvb1/log/add_first_layer.log
exp/multi_a/tdnn_7k_rvb1/log/init_mdl.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.0.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.0.log
exp/multi_a/tdnn_7k_rvb1/log/train.0.1.log
exp/multi_a/tdnn_7k_rvb1/log/train.0.2.log
exp/multi_a/tdnn_7k_rvb1/log/train.0.3.log
exp/multi_a/tdnn_7k_rvb1/log/train.0.4.log
exp/multi_a/tdnn_7k_rvb1/log/select.0.log
exp/multi_a/tdnn_7k_rvb1/log/progress.1.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.1.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.1.log
exp/multi_a/tdnn_7k_rvb1/log/train.1.1.log
exp/multi_a/tdnn_7k_rvb1/log/train.1.4.log
exp/multi_a/tdnn_7k_rvb1/log/train.1.2.log
```
```
ls exp/multi_a/tdnn_7k_rvb1/log/* -tr | tail -20
exp/multi_a/tdnn_7k_rvb1/log/train.1454.1.log
exp/multi_a/tdnn_7k_rvb1/log/train.1454.8.log
exp/multi_a/tdnn_7k_rvb1/log/average.1454.log
exp/multi_a/tdnn_7k_rvb1/log/progress.1455.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.1454.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.1454.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.4.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.3.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.6.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.5.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.1.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.2.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.7.log
exp/multi_a/tdnn_7k_rvb1/log/train.1455.8.log
exp/multi_a/tdnn_7k_rvb1/log/average.1455.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.1455.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.1455.log
exp/multi_a/tdnn_7k_rvb1/log/combine.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_train.final.log
exp/multi_a/tdnn_7k_rvb1/log/compute_prob_valid.final.log
```

## make_phone_lm.log

```
# gunzip -c exp/multi_a/tree_tdnn_7k_rvb1/ali.1.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.2.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.3.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.4.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.5.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.6.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.7.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.8.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.9.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.10.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.11.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.12.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.13.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.14.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.15.gz exp/multi_a/tree_tdnn_7k_rvb1/ali.16.gz | ali-to-phones exp/multi_a/tree_tdnn_7k_rvb1/final.mdl ark:- ark:- | chain-est-phone-lm --num-extra-lm-states=2000 ark:- exp/multi_a/tdnn_7k_rvb1/phone_lm.fst 
# Started at Wed May  2 00:49:53 KST 2018
#
ali-to-phones exp/multi_a/tree_tdnn_7k_rvb1/final.mdl ark:- ark:- 
chain-est-phone-lm --num-extra-lm-states=2000 ark:- exp/multi_a/tdnn_7k_rvb1/phone_lm.fst 
LOG (chain-est-phone-lm[5.2.107~2-e892]:main():chain-est-phone-lm.cc:62) Reading phone sequences
LOG (ali-to-phones[5.2.107~2-e892]:main():ali-to-phones.cc:134) Done 2376835 utterances.
LOG (chain-est-phone-lm[5.2.107~2-e892]:main():chain-est-phone-lm.cc:67) Estimating phone LM
LOG (chain-est-phone-lm[5.2.107~2-e892]:Estimate():language-model.cc:299) Estimating language model with --no-prune-ngram-order=3, --ngram-order=4, --num-extra-lm-state=2000
LOG (chain-est-phone-lm[5.2.107~2-e892]:DoBackoff():language-model.cc:253) In LM [hard] backoff, target num states was 10133 + --num-extra-lm-states=2000 = 12133, pruned from 156451 to 12133
LOG (chain-est-phone-lm[5.2.107~2-e892]:OutputToFst():language-model.cc:395) Total number of phone instances seen was 111289332
LOG (chain-est-phone-lm[5.2.107~2-e892]:OutputToFst():language-model.cc:396) Perplexity on training data is: 7.00846
LOG (chain-est-phone-lm[5.2.107~2-e892]:OutputToFst():language-model.cc:397) Note: perplexity on unseen data will be infinity as there is no smoothing.  This is by design, to reduce the number of arcs.
LOG (chain-est-phone-lm[5.2.107~2-e892]:OutputToFst():language-model.cc:405) Created phone language model with 12133 states and 195194 arcs.
LOG (chain-est-phone-lm[5.2.107~2-e892]:main():chain-est-phone-lm.cc:73) Estimated phone language model and wrote it to exp/multi_a/tdnn_7k_rvb1/phone_lm.fst
# Accounting: time=64 threads=1
# Ended (code 0) at Wed May  2 00:50:57 KST 2018, elapsed time 64 seconds
```

##

```
# chain-make-den-fst exp/multi_a/tdnn_7k_rvb1/tree exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/phone_lm.fst exp/multi_a/tdnn_7k_rvb1/den.fst exp/multi_a/tdnn_7k_rvb1/normalization.fst 
# Started at Wed May  2 00:50:57 KST 2018
#
chain-make-den-fst exp/multi_a/tdnn_7k_rvb1/tree exp/multi_a/tdnn_7k_rvb1/0.trans_mdl exp/multi_a/tdnn_7k_rvb1/phone_lm.fst exp/multi_a/tdnn_7k_rvb1/den.fst exp/multi_a/tdnn_7k_rvb1/normalization.fst 
LOG (chain-make-den-fst[5.2.107~2-e892]:CreateDenominatorFst():chain-den-graph.cc:306) Number of states and arcs in phone-LM FST is 12133 and 195194
LOG (chain-make-den-fst[5.2.107~2-e892]:CreateDenominatorFst():chain-den-graph.cc:328) Number of states and arcs in context-dependent LM FST is 12201 and 196617
LOG (chain-make-den-fst[5.2.107~2-e892]:CreateDenominatorFst():chain-den-graph.cc:361) Number of states and arcs in transition-id FST is 99616 and 371447
LOG (chain-make-den-fst[5.2.107~2-e892]:CreateDenominatorFst():chain-den-graph.cc:369) Number of states and arcs in transition-id FST after removing epsilons is 87416 and 2038072
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:228) Number of states and arcs in transition-id FST after reversed minimization is 72553 and 1851031 (pass 1)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:233) Number of states and arcs in transition-id FST after regular minimization is 58403 and 1637384 (pass 1)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:228) Number of states and arcs in transition-id FST after reversed minimization is 58405 and 1637386 (pass 2)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:233) Number of states and arcs in transition-id FST after regular minimization is 58287 and 1636513 (pass 2)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:228) Number of states and arcs in transition-id FST after reversed minimization is 58289 and 1636515 (pass 3)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:233) Number of states and arcs in transition-id FST after regular minimization is 58247 and 1636212 (pass 3)
LOG (chain-make-den-fst[5.2.107~2-e892]:DenGraphMinimizeWrapper():chain-den-graph.cc:238) Number of states and arcs in transition-id FST after removing any epsilons introduced by reversal is 58241 and 1635935
LOG (chain-make-den-fst[5.2.107~2-e892]:PrintDenGraphStats():chain-den-graph.cc:269) Number of states is 58241 and arcs 1635935; number of states with in-degree <= 3 is 5187 and with out-degree <= 3 is 3054
LOG (chain-make-den-fst[5.2.107~2-e892]:main():chain-make-den-fst.cc:78) Write denominator FST to exp/multi_a/tdnn_7k_rvb1/den.fst and normalization FST to exp/multi_a/tdnn_7k_rvb1/normalization.fst
# Accounting: time=26 threads=1
# Ended (code 0) at Wed May  2 00:51:23 KST 2018, elapsed time 26 seconds
```

## nnet_init.log

```
# nnet3-init --srand=-2 exp/multi_a/tdnn_7k_rvb1/configs/init.config exp/multi_a/tdnn_7k_rvb1/init.raw 
# Started at Wed May  2 00:51:23 KST 2018
#
nnet3-init --srand=-2 exp/multi_a/tdnn_7k_rvb1/configs/init.config exp/multi_a/tdnn_7k_rvb1/init.raw 
LOG (nnet3-init[5.2.107~2-e892]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/multi_a/tdnn_7k_rvb1/init.raw
# Accounting: time=0 threads=1
# Ended (code 0) at Wed May  2 00:51:23 KST 2018, elapsed time 0 seconds
```

## get_lda_stats.9.log

```
# nnet3-chain-acc-lda-stats --rand-prune=4.0 exp/multi_a/tdnn_7k_rvb1/init.raw ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.9.ark exp/multi_a/tdnn_7k_rvb1/9.lda_stats 
# Started at Wed May  2 03:43:35 KST 2018
#
nnet3-chain-acc-lda-stats --rand-prune=4.0 exp/multi_a/tdnn_7k_rvb1/init.raw ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.9.ark exp/multi_a/tdnn_7k_rvb1/9.lda_stats 
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:main():nnet3-chain-acc-lda-stats.cc:195) Processed 10315 examples.
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:WriteStats():nnet3-chain-acc-lda-stats.cc:67) Accumulated stats, soft frame count = 515444.  Wrote to exp/multi_a/tdnn_7k_rvb1/9.lda_stats
LOG (nnet3-chain-acc-lda-stats[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.00958 seconds taken in nnet3 compilation total (breakdown: 0.000521 compilation, 7.9e-05 optimization, 0 shortcut expansion, 4.7e-05 checking, 4.94e-06 computing indexes, 0.00893 misc.)
# Accounting: time=4 threads=1
# Ended (code 0) at Wed May  2 03:43:39 KST 2018, elapsed time 4 seconds
```

## sum_transform_stats.log

```
# sum-lda-accs exp/multi_a/tdnn_7k_rvb1/lda_stats exp/multi_a/tdnn_7k_rvb1/1.lda_stats exp/multi_a/tdnn_7k_rvb1/2.lda_stats exp/multi_a/tdnn_7k_rvb1/3.lda_stats exp/multi_a/tdnn_7k_rvb1/4.lda_stats exp/multi_a/tdnn_7k_rvb1/5.lda_stats exp/multi_a/tdnn_7k_rvb1/6.lda_stats exp/multi_a/tdnn_7k_rvb1/7.lda_stats exp/multi_a/tdnn_7k_rvb1/8.lda_stats exp/multi_a/tdnn_7k_rvb1/9.lda_stats exp/multi_a/tdnn_7k_rvb1/10.lda_stats 
# Started at Wed May  2 03:43:59 KST 2018
#
sum-lda-accs exp/multi_a/tdnn_7k_rvb1/lda_stats exp/multi_a/tdnn_7k_rvb1/1.lda_stats exp/multi_a/tdnn_7k_rvb1/2.lda_stats exp/multi_a/tdnn_7k_rvb1/3.lda_stats exp/multi_a/tdnn_7k_rvb1/4.lda_stats exp/multi_a/tdnn_7k_rvb1/5.lda_stats exp/multi_a/tdnn_7k_rvb1/6.lda_stats exp/multi_a/tdnn_7k_rvb1/7.lda_stats exp/multi_a/tdnn_7k_rvb1/8.lda_stats exp/multi_a/tdnn_7k_rvb1/9.lda_stats exp/multi_a/tdnn_7k_rvb1/10.lda_stats 
# Accounting: time=1 threads=1
# Ended (code 0) at Wed May  2 03:44:00 KST 2018, elapsed time 1 seconds
```

## get_transform.log

```
# nnet-get-feature-transform exp/multi_a/tdnn_7k_rvb1/lda.mat exp/multi_a/tdnn_7k_rvb1/lda_stats 
# Started at Wed May  2 03:44:00 KST 2018
#
nnet-get-feature-transform exp/multi_a/tdnn_7k_rvb1/lda.mat exp/multi_a/tdnn_7k_rvb1/lda_stats 
LOG (nnet-get-feature-transform[5.2.107~2-e892]:Estimate():get-feature-transform.cc:35) Data count is 5.15822e+06
LOG (nnet-get-feature-transform[5.2.107~2-e892]:EstimateInternal():get-feature-transform.cc:84) LDA singular values are  [ 1.22767 0.769907 0.30446 0.257567 0.180265 0.132707 0.127364 0.1136 0.0724881 0.0587797 0.0567012 0.038258 0.0353782 0.0305035 0.0292075 0.0275438 0.0239563 0.0201596 0.01909 0.0182027 0.0174205 0.0166331 0.0164506 0.0162405 0.0160173 0.0158556 0.0156867 0.015654 0.0155085 0.0153507 0.0152542 0.0151491 0.0151362 0.0148654 0.0148084 0.0147516 0.0146876 0.0146638 0.0145834 0.0144737 0.0144498 0.0143937 0.0143235 0.0142838 0.0142188 0.0141283 0.0141063 0.014057 0.014001 0.0139835 0.0138708 0.0138393 0.0137755 0.0137545 0.0137245 0.0136988 0.0136058 0.0135645 0.0135361 0.0134715 0.0134344 0.0133786 0.0133145 0.0132909 0.0132489 0.0132265 0.0131612 0.0130978 0.0130682 0.0129866 0.0129788 0.0128807 0.0128358 0.0128038 0.0127779 0.0127265 0.0127043 0.0126714 0.0126596 0.0126459 0.0126166 0.0125597 0.0124631 0.0124383 0.0123743 0.0123666 0.0123205 0.012292 0.0122332 0.012197 0.0121513 0.0121211 0.0121076 0.0120605 0.0120159 0.0119371 0.0119112 0.0118926 0.0118403 0.0117701 0.0117262 0.0116893 0.0116711 0.0116029 0.011587 0.0115262 0.0115162 0.0114608 0.0114173 0.0113898 0.0113694 0.011313 0.0112794 0.0112748 0.0112097 0.0111389 0.0111096 0.0110704 0.0110064 0.0109763 0.0109067 0.0108798 0.0108129 0.0107894 0.010774 0.0107475 0.0106605 0.0106387 0.0105921 0.0105755 0.0104825 0.0104465 0.0103819 0.0103154 0.0102656 0.010178 0.0101307 0.0101013 0.0100881 0.0100791 0.010013 0.00995392 0.00987662 0.00983519 0.0097972 0.00975593 0.00970295 0.00954618 0.00945659 0.00941827 0.0093879 0.00936638 0.00929314 0.0092598 0.00914529 0.00909925 0.00907747 0.0090417 0.00899492 0.00892686 0.00890253 0.00885664 0.00883876 0.00879974 0.00878701 0.00871172 0.00869657 0.00863877 0.00861454 0.00856676 0.0085303 0.00851104 0.00850293 0.00844828 0.00840691 0.00839142 0.00836152 0.00832563 0.00827521 0.008245 0.00820952 0.00818136 0.0081601 0.00810169 0.00803684 0.00802254 0.0080163 0.0079756 0.00795834 0.00793003 0.00791504 0.0078926 0.00783523 0.00780568 0.00780227 0.00776975 0.00773221 0.00770822 0.0076486 0.00762449 0.00759192 0.00754767 0.00752766 0.00749049 0.00744637 0.00743124 0.00736803 0.00730316 0.00727787 0.0072566 0.00723222 0.00717549 0.00710938 0.00706796 0.00704194 0.00694726 0.00689571 0.0068306 0.00676163 0.00669622 ]
LOG (nnet-get-feature-transform[5.2.107~2-e892]:EstimateInternal():get-feature-transform.cc:86) Sum of all singular values is 5.73521
LOG (nnet-get-feature-transform[5.2.107~2-e892]:EstimateInternal():get-feature-transform.cc:87) Sum of selected singular values is 5.73521
# Accounting: time=0 threads=1
# Ended (code 0) at Wed May  2 03:44:00 KST 2018, elapsed time 0 seconds
```

## add_first_layer.log

```
# nnet3-init --srand=-1 exp/multi_a/tdnn_7k_rvb1/init.raw exp/multi_a/tdnn_7k_rvb1/configs/final.config exp/multi_a/tdnn_7k_rvb1/0.raw 
# Started at Wed May  2 03:44:01 KST 2018
#
nnet3-init --srand=-1 exp/multi_a/tdnn_7k_rvb1/init.raw exp/multi_a/tdnn_7k_rvb1/configs/final.config exp/multi_a/tdnn_7k_rvb1/0.raw 
LOG (nnet3-init[5.2.107~2-e892]:main():nnet3-init.cc:68) Read raw neural net from exp/multi_a/tdnn_7k_rvb1/init.raw
LOG (nnet3-init[5.2.107~2-e892]:main():nnet3-init.cc:80) Initialized raw neural net and wrote it to exp/multi_a/tdnn_7k_rvb1/0.raw
# Accounting: time=1 threads=1
# Ended (code 0) at Wed May  2 03:44:02 KST 2018, elapsed time 1 seconds
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

## train.1.1.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/train.1.1.log
# nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --write-cache=exp/multi_a/tdnn_7k_rvb1/cache.2 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 "nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.5.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |" exp/multi_a/tdnn_7k_rvb1/2.1.raw 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --write-cache=exp/multi_a/tdnn_7k_rvb1/cache.2 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 'nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.5.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |' exp/multi_a/tdnn_7k_rvb1/2.1.raw 
LOG (nnet3-chain-train[5.2.107~2-e892]:IsComputeExclusive():cu-device.cc:263) CUDA setup operating under Compute Exclusive Process Mode.
LOG (nnet3-chain-train[5.2.107~2-e892]:FinalizeActiveGpu():cu-device.cc:225) The active GPU is [2]: GeForce GTX TITAN X	free:12067M, used:139M, total:12207M, free/total:0.988572 version 5.2
nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
LOG (nnet3-chain-train[5.2.107~2-e892]:NnetChainTrainer():nnet-chain-training.cc:52) Read computation cache from exp/multi_a/tdnn_7k_rvb1/cache.1
nnet3-chain-merge-egs --minibatch-size=128 ark:- ark:- 
nnet3-chain-shuffle-egs --buffer-size=5000 --srand=1 ark:- ark:- 
nnet3-chain-copy-egs --frame-shift=2 ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.5.ark ark:- 
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16598 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682797 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.169122 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682555 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158501 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682904 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161801 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681415 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160588 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681422 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152283 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682185 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153924 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68155 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149353 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681794 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149117 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681703 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149388 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681835 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 0-9 is -7.16759 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 0-9 is -0.57137 + -0.00545086 = -0.576821 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154133 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682825 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163343 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682745 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163482 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681927 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.165359 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681897 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148251 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682001 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.169869 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682634 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160362 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682338 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15287 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681463 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150918 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68203 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154036 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681954 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 10-19 is -6.99532 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 10-19 is -0.560515 + -0.00617849 = -0.566693 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159461 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682742 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15915 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682018 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145971 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681634 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152219 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681322 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147991 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681791 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145135 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68127 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149166 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681569 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14535 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681392 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152159 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68142 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157242 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681722 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 20-29 is -6.82259 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 20-29 is -0.544653 + -0.00681565 = -0.551469 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146484 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681797 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154309 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68093 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.136191 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680257 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148172 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681323 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158367 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681684 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154839 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68168 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143939 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680776 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154731 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680902 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148072 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681057 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149715 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681197 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 30-39 is -6.60965 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 30-39 is -0.515644 + -0.00737702 = -0.523021 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15682 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681575 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150477 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681052 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16096 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681059 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148931 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68104 with max-change=2.
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 10357 neural-network training examples, wrote 10357
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146895 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681137 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14582 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681056 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148045 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681095 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154062 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680657 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.131296 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680655 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163692 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68156 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 40-49 is -6.4448 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 40-49 is -0.50039 + -0.00802316 = -0.508413 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158072 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68089 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148991 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680959 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144921 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680935 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153478 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680556 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16176 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68105 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153478 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679915 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142591 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680594 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149569 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680621 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147786 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.6801 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145633 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680174 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 50-59 is -6.27838 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 50-59 is -0.488802 + -0.00855163 = -0.497354 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143469 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680655 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144223 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679213 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144886 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680948 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150299 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680138 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148551 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679781 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150623 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680063 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147603 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680554 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156547 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681193 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.141785 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68029 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14833 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680168 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 60-69 is -6.1251 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 60-69 is -0.474288 + -0.00910429 = -0.483393 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149245 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680075 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149931 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680704 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156209 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680435 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142301 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680977 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144318 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680428 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148669 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680403 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150094 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680185 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150743 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680342 with max-change=2.
LOG (nnet3-chain-shuffle-egs[5.2.107~2-e892]:main():nnet3-chain-shuffle-egs.cc:104) Shuffled order of 10357 neural-network training examples using a buffer (partial randomization)
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={128->80,d=117}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 10357 egs of avg. size 181 into 80 minibatches, discarding 1.13% of egs.  Avg minibatch size was 128, #distinct types of egs/minibatches was 1/1
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158083 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680294 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146879 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680015 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:295) Overall average objective function for 'output' is -0.516688 + -0.00764069 = -0.524329 over 512000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-0.516688
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:292) Overall average objective function for 'output-xent' is -6.56282 over 512000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-6.56282
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn1.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn2.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn3.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn4.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn5.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn6.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For prefinal-chain.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output-xent.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:260) The global max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:~NnetChainTrainer():nnet-chain-training.cc:272) Wrote computation cache to exp/multi_a/tdnn_7k_rvb1/cache.2
LOG (nnet3-chain-train[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.0384 seconds taken in nnet3 compilation total (breakdown: 0.0199 compilation, 0.0026 optimization, 0.00502 shortcut expansion, 0.000739 checking, 0.00161 computing indexes, 0.00854 misc.)
LOG (nnet3-chain-train[5.2.107~2-e892]:main():nnet3-chain-train.cc:97) Wrote raw model to exp/multi_a/tdnn_7k_rvb1/2.1.raw
# Accounting: time=89 threads=1
# Ended (code 0) at Wed May  2 03:47:14 KST 2018, elapsed time 89 seconds
```

## train.1.4.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/train.1.4.log
# nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 "nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.8.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |" exp/multi_a/tdnn_7k_rvb1/2.4.raw 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 'nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.8.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |' exp/multi_a/tdnn_7k_rvb1/2.4.raw 
LOG (nnet3-chain-train[5.2.107~2-e892]:IsComputeExclusive():cu-device.cc:263) CUDA setup operating under Compute Exclusive Process Mode.
LOG (nnet3-chain-train[5.2.107~2-e892]:FinalizeActiveGpu():cu-device.cc:225) The active GPU is [3]: GeForce GTX TITAN X	free:12067M, used:139M, total:12207M, free/total:0.988572 version 5.2
nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
LOG (nnet3-chain-train[5.2.107~2-e892]:NnetChainTrainer():nnet-chain-training.cc:52) Read computation cache from exp/multi_a/tdnn_7k_rvb1/cache.1
nnet3-chain-copy-egs --frame-shift=2 ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.8.ark ark:- 
nnet3-chain-merge-egs --minibatch-size=128 ark:- ark:- 
nnet3-chain-shuffle-egs --buffer-size=5000 --srand=1 ark:- ark:- 
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.170458 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682838 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152405 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681333 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156838 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68214 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159816 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682115 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155409 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682036 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158911 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682349 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149874 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682318 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153963 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.683191 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159153 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682502 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148075 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681993 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 0-9 is -7.23949 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 0-9 is -0.589193 + -0.00543157 = -0.594625 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160376 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681998 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151738 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681275 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150656 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681327 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.168716 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681659 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153748 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681435 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153697 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681328 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151741 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681859 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151223 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681789 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147729 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681261 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159181 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682526 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 10-19 is -6.97438 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 10-19 is -0.560456 + -0.00609746 = -0.566553 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147113 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681517 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148152 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68125 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14968 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682243 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163233 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682015 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159722 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681244 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146072 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681293 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15247 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681069 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161587 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68147 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159552 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682762 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157972 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682242 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 20-29 is -6.84202 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 20-29 is -0.549764 + -0.00674821 = -0.556512 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156916 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682078 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15055 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681669 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149908 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680614 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156002 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681546 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156598 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681796 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155769 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681317 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152068 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680958 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148393 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681776 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152804 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681752 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143736 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681087 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 30-39 is -6.66236 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 30-39 is -0.527726 + -0.00742017 = -0.535146 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14814 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681182 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160364 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681927 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151593 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681587 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153072 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681095 with max-change=2.
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 10386 neural-network training examples, wrote 10386
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152082 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680524 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145233 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681003 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154075 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681228 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149182 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680054 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157779 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681444 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155714 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681497 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 40-49 is -6.46426 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 40-49 is -0.503743 + -0.00798657 = -0.51173 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147717 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681141 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14683 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680173 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148767 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679666 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145416 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681471 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.1512 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680515 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152005 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681221 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154928 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680538 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150107 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680968 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150178 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680436 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155891 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679603 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 50-59 is -6.27693 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 50-59 is -0.486504 + -0.0085983 = -0.495103 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15195 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680908 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.138029 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680199 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151598 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680529 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15228 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680483 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155604 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681039 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158442 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68069 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152213 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68069 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154416 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681164 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144055 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679805 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151236 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680358 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 60-69 is -6.19987 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 60-69 is -0.488866 + -0.00922385 = -0.49809 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146025 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680322 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14758 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680264 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149181 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680565 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145066 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679986 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144408 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680133 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144304 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680951 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146944 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679878 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.139787 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679236 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145967 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679863 with max-change=2.
LOG (nnet3-chain-shuffle-egs[5.2.107~2-e892]:main():nnet3-chain-shuffle-egs.cc:104) Shuffled order of 10386 neural-network training examples using a buffer (partial randomization)
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={128->81,d=18}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 10386 egs of avg. size 181 into 81 minibatches, discarding 0.1733% of egs.  Avg minibatch size was 128, #distinct types of egs/minibatches was 1/1
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145244 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680898 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 70-79 is -6.045 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 70-79 is -0.479339 + -0.00970162 = -0.48904 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150558 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68036 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:295) Overall average objective function for 'output' is -0.522694 + -0.00767906 = -0.530373 over 518400 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-0.522694
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:292) Overall average objective function for 'output-xent' is -6.5812 over 518400 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-6.5812
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn1.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn2.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn3.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn4.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn5.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn6.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For prefinal-chain.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output-xent.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:260) The global max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.0256 seconds taken in nnet3 compilation total (breakdown: 0.012 compilation, 0.00153 optimization, 0.00288 shortcut expansion, 0.000452 checking, 0.00143 computing indexes, 0.00732 misc.)
LOG (nnet3-chain-train[5.2.107~2-e892]:main():nnet3-chain-train.cc:97) Wrote raw model to exp/multi_a/tdnn_7k_rvb1/2.4.raw
# Accounting: time=93 threads=1
# Ended (code 0) at Wed May  2 03:47:18 KST 2018, elapsed time 93 seconds
```

## train.1.2.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/train.1.2.log
# nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 "nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=0                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.6.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |" exp/multi_a/tdnn_7k_rvb1/2.2.raw 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 'nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs                         --frame-shift=0                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.6.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |' exp/multi_a/tdnn_7k_rvb1/2.2.raw 
LOG (nnet3-chain-train[5.2.107~2-e892]:IsComputeExclusive():cu-device.cc:263) CUDA setup operating under Compute Exclusive Process Mode.
LOG (nnet3-chain-train[5.2.107~2-e892]:FinalizeActiveGpu():cu-device.cc:225) The active GPU is [0]: GeForce GTX TITAN X	free:12067M, used:139M, total:12207M, free/total:0.988572 version 5.2
nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
LOG (nnet3-chain-train[5.2.107~2-e892]:NnetChainTrainer():nnet-chain-training.cc:52) Read computation cache from exp/multi_a/tdnn_7k_rvb1/cache.1
nnet3-chain-merge-egs --minibatch-size=128 ark:- ark:- 
nnet3-chain-shuffle-egs --buffer-size=5000 --srand=1 ark:- ark:- 
nnet3-chain-copy-egs --frame-shift=0 ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.6.ark ark:- 
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.170256 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682961 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16292 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682216 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154899 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681776 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.167377 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682399 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163121 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682296 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158368 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682096 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.167012 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682257 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.164503 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682467 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.170715 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682098 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160767 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.683074 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 0-9 is -7.20196 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 0-9 is -0.579468 + -0.00545549 = -0.584923 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159849 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682311 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158495 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682062 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160716 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682858 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160898 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682433 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156697 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681977 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154763 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682264 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.162416 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682581 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146546 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681503 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149742 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68205 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157058 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682095 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 10-19 is -6.99759 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 10-19 is -0.556921 + -0.00618028 = -0.563101 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15537 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682117 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.1503 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680712 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15877 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682303 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160687 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682186 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159765 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682033 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152911 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681885 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156428 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68195 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153785 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681167 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158006 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682098 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151044 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681165 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 20-29 is -6.81323 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 20-29 is -0.538119 + -0.0068262 = -0.544945 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157997 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681871 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155005 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681803 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156028 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681607 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150695 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681522 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159295 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682033 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144839 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681562 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158134 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68151 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15622 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681451 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146969 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680359 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153005 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681788 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 30-39 is -6.65857 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 30-39 is -0.527103 + -0.00749154 = -0.534594 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154657 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680294 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16044 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681675 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157059 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681701 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149319 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681762 with max-change=2.
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 10357 neural-network training examples, wrote 10357
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161968 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682339 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144196 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681461 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142588 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681113 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16013 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681823 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155896 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682027 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157492 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681988 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 40-49 is -6.47031 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 40-49 is -0.511091 + -0.00812112 = -0.519212 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152638 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681686 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146898 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680303 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146394 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680903 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155414 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68187 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14596 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680295 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144333 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680875 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149926 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680012 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147447 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680838 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146272 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680487 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144063 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680127 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 50-59 is -6.28228 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 50-59 is -0.486056 + -0.00867426 = -0.494731 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143384 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68089 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150323 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68058 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145675 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680644 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154757 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681395 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148109 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680752 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.1553 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679772 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151693 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680511 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14935 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680235 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.133956 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679705 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152058 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680639 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 60-69 is -6.16567 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 60-69 is -0.475523 + -0.00918538 = -0.484708 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146425 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681322 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.134627 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.67965 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148445 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680294 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147715 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68085 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148376 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679534 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149165 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680663 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151217 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680187 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143684 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680275 with max-change=2.
LOG (nnet3-chain-shuffle-egs[5.2.107~2-e892]:main():nnet3-chain-shuffle-egs.cc:104) Shuffled order of 10357 neural-network training examples using a buffer (partial randomization)
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={128->80,d=117}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 10357 egs of avg. size 181 into 80 minibatches, discarding 1.13% of egs.  Avg minibatch size was 128, #distinct types of egs/minibatches was 1/1
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.134513 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680319 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148328 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680095 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:295) Overall average objective function for 'output' is -0.517885 + -0.00770921 = -0.525594 over 512000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-0.517885
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:292) Overall average objective function for 'output-xent' is -6.57684 over 512000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-6.57684
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn1.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn2.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn3.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn4.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn5.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn6.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For prefinal-chain.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output-xent.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:260) The global max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.036 seconds taken in nnet3 compilation total (breakdown: 0.0172 compilation, 0.00253 optimization, 0.00402 shortcut expansion, 0.000733 checking, 0.00309 computing indexes, 0.00848 misc.)
LOG (nnet3-chain-train[5.2.107~2-e892]:main():nnet3-chain-train.cc:97) Wrote raw model to exp/multi_a/tdnn_7k_rvb1/2.2.raw
# Accounting: time=94 threads=1
# Ended (code 0) at Wed May  2 03:47:19 KST 2018, elapsed time 94 seconds
```

## train.1.3.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/train.1.3.log
# nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 "nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |" exp/multi_a/tdnn_7k_rvb1/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=1                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.7.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |" exp/multi_a/tdnn_7k_rvb1/2.3.raw 
# Started at Wed May  2 03:45:45 KST 2018
#
nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --read-cache=exp/multi_a/tdnn_7k_rvb1/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=2.0 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --srand=1 'nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - |' exp/multi_a/tdnn_7k_rvb1/den.fst 'ark,bg:nnet3-chain-copy-egs                         --frame-shift=1                         ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.7.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=1 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=128 ark:- ark:- |' exp/multi_a/tdnn_7k_rvb1/2.3.raw 
LOG (nnet3-chain-train[5.2.107~2-e892]:IsComputeExclusive():cu-device.cc:263) CUDA setup operating under Compute Exclusive Process Mode.
LOG (nnet3-chain-train[5.2.107~2-e892]:FinalizeActiveGpu():cu-device.cc:225) The active GPU is [1]: GeForce GTX TITAN X	free:12067M, used:139M, total:12207M, free/total:0.988572 version 5.2
nnet3-am-copy --raw=true --learning-rate=0.00399578503349 --scale=1.0 exp/multi_a/tdnn_7k_rvb1/1.mdl - 
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:140) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to raw format as -
LOG (nnet3-chain-train[5.2.107~2-e892]:NnetChainTrainer():nnet-chain-training.cc:52) Read computation cache from exp/multi_a/tdnn_7k_rvb1/cache.1
nnet3-chain-merge-egs --minibatch-size=128 ark:- ark:- 
nnet3-chain-shuffle-egs --buffer-size=5000 --srand=1 ark:- ark:- 
nnet3-chain-copy-egs --frame-shift=1 ark:exp/multi_a/tdnn_7k_rvb1/egs/cegs.7.ark ark:- 
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.173862 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682728 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163884 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682699 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.164408 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682792 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.165892 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682524 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160017 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682812 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.147429 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682313 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.162708 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682492 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150145 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682001 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.163207 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.683024 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151147 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682114 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 0-9 is -7.20423 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 0-9 is -0.585552 + -0.00541245 = -0.590964 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151949 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68232 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.162617 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682353 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144666 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681714 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.16696 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682396 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15722 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682016 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156687 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681927 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.162347 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681777 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155948 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682068 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145803 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681037 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146621 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681698 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 10-19 is -7.02079 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 10-19 is -0.564639 + -0.00608194 = -0.570721 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.167102 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682011 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.155303 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680987 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.158419 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681483 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160129 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681902 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161036 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682832 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.159561 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682904 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161802 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.682414 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154335 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681686 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143852 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681652 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15334 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681004 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 20-29 is -6.84234 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 20-29 is -0.54726 + -0.00681047 = -0.55407 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152526 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681711 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150854 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681786 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146824 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681473 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.156613 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68203 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.150738 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681426 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151361 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681278 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.136467 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68068 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144916 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680505 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157792 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68181 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145881 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680426 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 30-39 is -6.65508 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 30-39 is -0.527558 + -0.00746951 = -0.535028 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146137 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680388 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153107 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681151 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154915 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681756 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161666 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681349 with max-change=2.
LOG (nnet3-chain-copy-egs[5.2.107~2-e892]:main():nnet3-chain-copy-egs.cc:346) Read 10387 neural-network training examples, wrote 10387
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151252 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681331 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15021 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681047 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14407 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680929 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.161272 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681802 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151889 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681223 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.137979 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680834 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 40-49 is -6.45622 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 40-49 is -0.504444 + -0.00807705 = -0.512521 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143747 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680926 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14548 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680504 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15605 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681179 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151699 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680845 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.140968 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680723 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142629 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680931 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.148934 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680678 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14217 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680722 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145765 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680835 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.160187 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680677 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 50-59 is -6.30729 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 50-59 is -0.489558 + -0.00867126 = -0.498229 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154503 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680713 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.140275 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680442 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.146719 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679713 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152859 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680224 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.152332 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681514 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.15154 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68099 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154637 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681171 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151088 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.681157 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142046 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680705 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.154378 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680861 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 60-69 is -6.13843 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 60-69 is -0.478045 + -0.00918777 = -0.487233 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.14811 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.67965 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.142639 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680142 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.143405 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68075 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153989 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680847 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.144551 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680589 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.141529 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.678981 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.153333 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.68064 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.157082 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680663 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.149307 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680889 with max-change=2.
LOG (nnet3-chain-shuffle-egs[5.2.107~2-e892]:main():nnet3-chain-shuffle-egs.cc:104) Shuffled order of 10387 neural-network training examples using a buffer (partial randomization)
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1125) Merged specific eg types as follows [format: <eg-size1>={<mb-size1>-><num-minibatches1>,<mbsize2>-><num-minibatches2>.../d=<num-discarded>},<egs-size2>={...},... (note,egs-size == number of input frames including context).
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintSpecificStats():nnet-example-utils.cc:1155) 181={128->81,d=19}
LOG (nnet3-chain-merge-egs[5.2.107~2-e892]:PrintAggregateStats():nnet-example-utils.cc:1121) Processed 10387 egs of avg. size 181 into 81 minibatches, discarding 0.1829% of egs.  Avg minibatch size was 128, #distinct types of egs/minibatches was 1/1
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.145074 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.679298 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:253) Average objective function for 'output-xent' for minibatches 70-79 is -6.03379 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintStatsForThisPhase():nnet-training.cc:271) Average objective function for 'output' for minibatches 70-79 is -0.473882 + -0.00974085 = -0.483623 over 64000 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:UpdateNnetWithMaxChange():nnet-utils.cc:1429) Per-component max-change active on 9 / 10 Updatable Components.(smallest factor=0.151965 on tdnn2.affine with max-change=0.75). Global max-change factor was 0.680014 with max-change=2.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:295) Overall average objective function for 'output' is -0.520422 + -0.00771075 = -0.528133 over 518400 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-0.520422
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:292) Overall average objective function for 'output-xent' is -6.57391 over 518400 frames.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintTotalStats():nnet-training.cc:299) [this line is to be parsed by a script:] log-prob-per-frame=-6.57391
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn1.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn2.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn3.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn4.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn5.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For tdnn6.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For prefinal-chain.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:249) For output-xent.affine, per-component max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:PrintMaxChangeStats():nnet-chain-training.cc:260) The global max-change was enforced 100 % of the time.
LOG (nnet3-chain-train[5.2.107~2-e892]:~CachingOptimizingCompiler():nnet-optimize.cc:670) 0.012 seconds taken in nnet3 compilation total (breakdown: 0 compilation, 0 optimization, 0.00324 shortcut expansion, 0 checking, 0.00117 computing indexes, 0.00757 misc.)
LOG (nnet3-chain-train[5.2.107~2-e892]:main():nnet3-chain-train.cc:97) Wrote raw model to exp/multi_a/tdnn_7k_rvb1/2.3.raw
# Accounting: time=95 threads=1
# Ended (code 0) at Wed May  2 03:47:20 KST 2018, elapsed time 95 seconds
```

## average.1.log

```
cat exp/multi_a/tdnn_7k_rvb1/log/average.1.log
# nnet3-average exp/multi_a/tdnn_7k_rvb1/2.1.raw exp/multi_a/tdnn_7k_rvb1/2.2.raw exp/multi_a/tdnn_7k_rvb1/2.3.raw exp/multi_a/tdnn_7k_rvb1/2.4.raw - | nnet3-am-copy --set-raw-nnet=- exp/multi_a/tdnn_7k_rvb1/1.mdl exp/multi_a/tdnn_7k_rvb1/2.mdl 
# Started at Wed May  2 03:47:20 KST 2018
#
nnet3-am-copy --set-raw-nnet=- exp/multi_a/tdnn_7k_rvb1/1.mdl exp/multi_a/tdnn_7k_rvb1/2.mdl 
nnet3-average exp/multi_a/tdnn_7k_rvb1/2.1.raw exp/multi_a/tdnn_7k_rvb1/2.2.raw exp/multi_a/tdnn_7k_rvb1/2.3.raw exp/multi_a/tdnn_7k_rvb1/2.4.raw - 
LOG (nnet3-average[5.2.107~2-e892]:main():nnet3-average.cc:110) Averaged parameters of 4 neural nets, and wrote to -
LOG (nnet3-am-copy[5.2.107~2-e892]:main():nnet3-am-copy.cc:147) Copied neural net from exp/multi_a/tdnn_7k_rvb1/1.mdl to exp/multi_a/tdnn_7k_rvb1/2.mdl
# Accounting: time=1 threads=1
# Ended (code 0) at Wed May  2 03:47:21 KST 2018, elapsed time 1 seconds
```
