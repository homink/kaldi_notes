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

In the training, TDNN network and input arguments are configured as follows:

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

We find log files in a timely manner and see what commands are executed in order. By listing list up log files, we can see the pattern of the chain training flow. One remarkable poing is [parallel training](https://arxiv.org/pdf/1410.7455.pdf) where the parameter-averaging performs with separate GPU training processes. Details of log files are browsed [here](https://github.com/homink/kaldi_notes/blob/master/chain_training_log.md).

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
...
train.181.3.log
train.181.2.log
train.181.1.log
train.181.4.log
average.181.log
progress.182.log
compute_prob_train.182.log
compute_prob_valid.182.log
train.182.2.log
train.182.5.log
train.182.1.log
train.182.4.log
train.182.3.log
average.182.log
progress.183.log
compute_prob_train.183.log
compute_prob_valid.183.log

...

train.545.5.log
train.545.1.log
train.545.2.log
train.545.4.log
train.545.3.log
average.545.log
progress.546.log
compute_prob_train.545.log
compute_prob_valid.545.log
train.546.4.log
train.546.2.log
train.546.3.log
train.546.1.log
train.546.5.log
train.546.6.log
average.546.log
progress.547.log
compute_prob_train.546.log
compute_prob_valid.546.log
train.547.5.log

...

train.909.3.log
train.909.4.log
train.909.2.log
train.909.5.log
train.909.6.log
train.909.1.log
average.909.log
progress.910.log
compute_prob_valid.910.log
train.910.3.log
train.910.5.log
train.910.7.log
train.910.4.log
train.910.1.log
train.910.6.log
train.910.2.log
average.910.log
progress.911.log
compute_prob_train.910.log
compute_prob_valid.911.log

...

train.1273.4.log
train.1273.3.log
train.1273.1.log
train.1273.5.log
train.1273.2.log
train.1273.7.log
train.1273.6.log
average.1273.log
progress.1274.log
compute_prob_train.1273.log
compute_prob_valid.1273.log
train.1274.1.log
train.1274.8.log
train.1274.5.log
train.1274.6.log
train.1274.7.log
train.1274.2.log
train.1274.4.log
train.1274.3.log
average.1274.log
compute_prob_train.1274.log
compute_prob_valid.1274.log
...
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

We can check how the overfitting would come up in accuracy.report as follows.

```
head exp/multi_a/tdnn_7k_rvb1/accuracy.report
%Iter   duration        train_loss      valid_loss      difference
0       98.0    -1.07591        -1.08398        -0.00807
1       95.0    -0.631919       -0.638957       -0.007038
2       95.0    -0.415988       -0.429965       -0.013977
3       98.0    -0.358987       -0.369345       -0.010358
4       94.0    -0.325841       -0.333205       -0.007364
5       94.0    -0.306459       -0.310971       -0.004512
6       96.0    -0.290208       -0.297254       -0.007046
7       95.0    -0.278244       -0.286121       -0.007877
8       94.0    -0.272537       -0.274912       -0.002375
```
