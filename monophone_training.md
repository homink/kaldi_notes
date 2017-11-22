# The Monophone Training Output Review

## Browsing the output folder

```
$ ls exp/mono
0.mdl      ali.6.gz                        fsts.17.gz
40.mdl     ali.7.gz                        fsts.18.gz
40.occs    ali.8.gz                        fsts.19.gz
ali.1.gz   ali.9.gz                        fsts.2.gz
ali.10.gz  cmvn_opts                       fsts.20.gz
ali.11.gz  decode_nosp_tgsmall_dev_clean   fsts.3.gz
ali.12.gz  decode_nosp_tgsmall_dev_other   fsts.4.gz
ali.13.gz  decode_nosp_tgsmall_test_clean  fsts.5.gz
ali.14.gz  decode_nosp_tgsmall_test_other  fsts.6.gz
ali.15.gz  final.mdl                       fsts.7.gz
ali.16.gz  final.occs                      fsts.8.gz
ali.17.gz  fsts.1.gz                       fsts.9.gz
ali.18.gz  fsts.10.gz                      graph_nosp_tgsmall
ali.19.gz  fsts.11.gz                      log
ali.2.gz   fsts.12.gz                      num_jobs
ali.20.gz  fsts.13.gz                      phones.txt
ali.3.gz   fsts.14.gz                      tree
ali.4.gz   fsts.15.gz
ali.5.gz   fsts.16.gz
```

## Why ali.x.gz and fsts.x.gz files have numbers up to 20?
WHen running [steps/train_mono.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh), --nj was given by 20. This is stored in exp/mono/num_jobs file
```
$ more exp/mono/num_jobs
20
```

## What does exp/mono/final.mdl has?
```
$ show-transitions exp/mono/phones.txt exp/mono/final.mdl exp/mono/final.occs | head -8
show-transitions exp/mono/phones.txt exp/mono/final.mdl exp/mono/final.occs
Transition-state 1: phone = SIL hmm-state = 0 pdf = 0
 Transition-id = 1 p = 0.9135 count of pdf = 73871 [self-loop]
 Transition-id = 2 p = 0.01 count of pdf = 73871 [0 -> 1]
 Transition-id = 3 p = 0.01 count of pdf = 73871 [0 -> 2]
 Transition-id = 4 p = 0.0665041 count of pdf = 73871 [0 -> 3]
Transition-state 2: phone = SIL hmm-state = 1 pdf = 1
 Transition-id = 5 p = 0.948111 count of pdf = 13524 [self-loop]
 Transition-id = 6 p = 0.01 count of pdf = 13524 [1 -> 2]
$ show-transitions exp/mono/phones.txt exp/mono/final.mdl exp/mono/final.occs | tail -8
show-transitions exp/mono/phones.txt exp/mono/final.mdl exp/mono/final.occs
 Transition-id = 2191 p = 0.75 count of pdf = 3046 [self-loop]
 Transition-id = 2192 p = 0.25 count of pdf = 3046 [0 -> 1]
Transition-state 1057: phone = EH2_S hmm-state = 1 pdf = 125
 Transition-id = 2193 p = 0.75 count of pdf = 4621 [self-loop]
 Transition-id = 2194 p = 0.25 count of pdf = 4621 [1 -> 2]
Transition-state 1058: phone = EH2_S hmm-state = 2 pdf = 126
 Transition-id = 2195 p = 0.75 count of pdf = 7236 [self-loop]
 Transition-id = 2196 p = 0.25 count of pdf = 7236 [2 -> 3]
```

## How does exp/mono/final.mdl work with others?
```
$ ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | utils/int2sym.pl -f 2- exp/mono/phones.txt | head -4
ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 SIL IH0_B F_E HH_B IY1_I D_E R_B AH1_I N_E AW1_B T_E AH0_B V_E T_B ER1_I N_I AH0_I P_E S_B IY1_I D_E HH_B IY1_E W_B UH1_I D_I AH0_I N_I T_E D_B R_I EH1_I S_E AH1_B P_E AH0_B N_I D_E T_B EY1_I K_E DH_B AH1_E B_B AH1_I G_I IY0_E T_B IH0_E G_B OW1_E F_B ER0_E M_B AO1_I R_E SIL
103-1241-0019 SIL AY1_S HH_B AE1_I D_E M_B EY1_I D_E AH1_B P_E M_B AY1_E M_B AY1_I N_I D_E SIL DH_B AH0_I T_E IH0_B F_E Y_B UW1_E D_B IH1_I D_I AH0_I N_E K_B AH1_I M_E F_B ER0_E M_B IY1_E T_B IH0_E N_B AY1_I T_E SIL
103-1241-0025 SIL IH1_B T_I S_E W_B ER1_I S_E DH_B AE1_I N_E SIL EH1_B N_I IY0_I TH_I IH2_I NG_E Y_B UW1_E K_B UH1_I D_E IH0_B M_I AE1_I JH_I AH0_I N_E SIL M_B IH1_I S_I IH0_I Z_E S_B P_I EH1_I N_I S_I ER0_E S_B EH1_I D_E IH0_B T_E W_B AH0_I Z_E W_B IH1_I K_I AH0_I D_E AH0_B V_E M_B IY1_E T_B IH0_E T_B AO1_I K_E L_B AY1_I K_E DH_B AE1_I T_E SIL
1034-121119-0009 SIL W_B AH0_I Z_E AH0_B S_I EH1_I N_I D_I IH0_I NG_E DH_B AH1_E S_B T_I EH1_I R_I Z_E L_B IY1_I D_I IH0_I NG_E T_B IH0_E SPN_S AH0_B P_I AA1_I R_I T_I M_I AH0_I N_I T_I S_E SIL
```
