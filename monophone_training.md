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
$ ali-to-pdf exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | utils/int2sym.pl -f 2- exp/mono/phones.txt | head -4
ali-to-pdf exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SH_I SH_I SH_S W_B AY0_I AY0_I AY0_I AY0_S AY1_B EY2_I EY2_S B_B OY_I OY_I OY_S OY_S OY_S OY0_B OY0_B OY0_B OY0_B OY0_B AO1_I AO1_S AO2_B CH_B CH_B CH_E CH_E CH_I CH_I EY_I EY_I EY_I EY_I EY_S EY0_B UW0_B UW0_E UW0_E UW0_I UW0_I EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY2_B EY2_B EY2_E UW_E UW_I UW_S EY_I EY_S EY0_B AY1_E AY1_I AY1_S AY1_S AY1_S AY1_S UW_E UW_E UW_E UW_I UW_I UW_I UW_I UW_S UW_S UW_S UW_S AY2_B AY2_B AY2_E AY2_E AY2_E AY2_E AY2_I UW0_B UW0_E UW0_I UW0_I UW0_I EY_I EY_S EY_S EY0_B EY0_B EY0_E EY0_E EY0_I EY0_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_B S_E OY_I OY_I OY_I OY_S OY_S OY_S OY_S OY_S OY_S OY_S OY0_B AO1_I AO1_S AO2_B EY2_I EY2_I EY2_S EY2_S B_B OY_I OY_I OY_S OY_S OY0_B OY0_B OY0_B K_E K_E K_I K_I K_I K_I K_S K_S AY_S AY0_B AY0_E AO1_I AO1_S AO2_B AO2_B EY_I EY_S EY0_B UW0_B UW0_B UW0_B UW0_E UW0_I UW_E UW_I UW_S AO1_I AO1_S AO2_B AO2_B CH_B CH_E CH_E CH_I OY0_E OY0_E OY0_I OY0_S OY0_S OY0_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_E S_E S_E EY_I EY_I EY_I EY_I EY_S EY0_B EY0_B EY0_E EY0_E EY0_I EY0_I EY0_I EY0_S EY_I EY_I EY_I EY_S EY0_B UW0_B UW0_E UW0_I UW0_I AO1_I AO1_S AO2_B UW_E UW_I UW_I UW_I UW_S Y_S Y_S Z_B Z_B Z_B Z_E Z_E UW0_S UW0_S UW0_S UW1_B UW1_B UW1_E UW1_E NG_B NG_B NG_B NG_E NG_I EY_I EY_S EY0_B Z_I Z_I Z_S Z_S Z_S Z_S Z_S AO_B AO_B AO_B EY_I EY_I EY_I EY_I EY_S EY_S EY0_B EY0_B AY_B AY_B AY_E AY_I AY_I AY_I OY_I OY_S OY_S OY_S OY_S OY0_B OY0_B OY0_B UW_E UW_E UW_E UW_I UW_I UW_S SH_I SH_S W_B W_B AY_B AY_E AY_E AY_I AY_I AY_I AY_I EY1_B EY1_B EY1_B EY1_B EY1_B EY1_E EY1_E EY1_I EY1_I AY0_I AY0_I AY0_I AY0_I AY0_I AY0_I AY0_S AY0_S AY0_S AY1_B AY2_B AY2_E AY2_I W_E W_E W_E W_E W_I W_I W_I W_S T_B T_E T_E T_E T_E T_I T_I T_I T_I T_I T_I T_I CH_B CH_E CH_I CH_I CH_I CH_I CH_I CH_I CH_I <eps> <eps> <eps> <eps> <eps> <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I
103-1241-0019 <eps> <eps> <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I T_S T_S T_S N_B N_B N_B N_E N_E EY2_I EY2_I EY2_I EY2_S EY2_S B_B B_B B_E B_I B_S B_S AO1_I AO1_I AO1_S AO2_B W_E W_I W_I W_I W_I W_I W_S W_S Y_S Z_B Z_B Z_B Z_B Z_E AO1_I AO1_S AO2_B EY_I EY_I EY_I EY_I EY_I EY_S EY0_B EY0_E EY0_E EY0_E EY0_E EY0_I EY0_I EY0_S W_E W_I W_S T_S N_B N_B N_E N_E N_E N_E N_E N_E W_E W_E W_E W_E W_E W_E W_I W_I W_S W_S T_S N_B N_B N_E N_E N_E N_E N_E N_E N_E N_E N_E N_E N_E UW0_B UW0_E UW0_I AO1_I AO1_S AO2_B AO2_B AO2_B AO2_B AO2_B <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I NG_B NG_B NG_E NG_I NG_I EY_I EY_S EY0_B UW_E UW_I UW_S SH_I SH_S SH_S W_B W_B AY0_I AY0_I AY0_I AY0_S AY1_B UW1_I UW1_S UW1_S UW2_B S_I S_I S_S UW_B UW_B AO1_I AO1_S AO1_S AO1_S AO2_B SH_I SH_I SH_S SH_S SH_S W_B W_B AO1_I AO1_S AO2_B AO2_B AO2_B EY_I EY_S EY0_B UW0_B UW0_E UW0_I UW0_I UW0_I UW0_S UW1_B UW1_E UW1_E UW1_E UW1_E UW1_E UW1_E UW1_E UW1_E EY_I EY_I EY_S EY0_B W_E W_E W_I W_I W_I W_I W_S AY0_I AY0_S AY0_S AY0_S AY0_S AY0_S AY1_B AY1_B AY2_B AY2_E AY2_I W_E W_E W_E W_I W_I W_S OY_I OY_I OY_S OY_S OY0_B OY0_B UW_E UW_E UW_E UW_I UW_I UW_I UW_S SH_I SH_S W_B UW0_B UW0_E UW0_E UW0_I UW0_I UW0_I UW0_I UW0_I T_S T_S N_B N_B N_B N_B N_B N_E N_E N_E N_E N_E N_E N_E N_E N_E N_E UW_E UW_E UW_E UW_E UW_I UW_S UW_S <eps> <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I
103-1241-0025 <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SH_I SH_I SH_I SH_S W_B UW_E UW_E UW_I UW_I UW_S SPN_S SPN_S S_B S_B S_B S_E S_E S_E S_E S_E K_E K_E K_I K_S AY2_B AY2_B AY2_B AY2_E AY2_E AY2_E AY2_I SPN_S SPN_S SPN_S SPN_S S_B S_E S_E S_E S_E S_E NG_B NG_B NG_E NG_I B_E B_E B_I B_S UW0_B UW0_B UW0_E UW0_E UW0_E UW0_I <eps> <eps> <eps> <eps> SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I OY0_E OY0_E OY0_I OY0_I OY0_S UW0_B UW0_E UW0_E UW0_I OY_I OY_S OY_S OY_S OY_S OY_S OY0_B OY0_B OY0_B CH_S CH_S OY_B OY_B OY_B OY_E OY_E SH_I SH_S W_B Y_B Y_E Y_I Y_I UW1_I UW1_S UW1_S UW1_S UW2_B UW2_B UW2_B S_I S_S UW_B UW0_S UW0_S UW1_B UW1_B UW1_B UW1_E UW1_E UW1_E UW1_E UW1_E AY_S AY0_B AY0_B AY0_B AY0_E AO1_I AO1_S AO2_B AO2_B AO2_B AO2_B SH_I SH_S W_B W_E W_I W_I W_S B_E B_E B_E B_E B_E B_E B_E B_E B_E B_I B_I B_I B_S B_S AO0_S AO0_S AO0_S AO1_B AO1_B AO1_B AO1_B AO1_E AO1_E EY_I EY_S EY0_B EY0_B EY0_B UW0_B UW0_B UW0_B UW0_E UW0_E UW0_E UW0_I <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL SIL_I SIL_I SIL_I SIL_I SIL_I W_E W_E W_I W_I W_S SH_I SH_S W_B SPN_S SPN_S SPN_S SPN_S S_B S_B S_E SH_I SH_I SH_S W_B UW2_E UW2_E UW2_E UW2_I UW2_S SPN_S SPN_S S_B S_B S_E S_E S_E EY0_E EY0_I EY0_I EY0_S OY0_E OY0_E OY0_I OY0_I OY0_S UW0_B UW0_B UW0_E UW0_E UW0_E UW0_E UW0_I UW0_I UW0_I SPN_S SPN_S S_B S_B S_E S_E AY2_B AY2_E AY2_I SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_B S_E OY0_E OY0_I OY0_S AO1_I AO1_S AO2_B SH_I SH_S W_B UW_E UW_I UW_S K_E K_I K_I K_S EY_I EY_S EY0_B UW2_E UW2_I UW2_I UW2_I UW2_S UW2_S UW2_S UW2_S UW2_S UW2_S K_E K_E K_I K_S SH_I SH_S W_B W_B UW0_S UW0_S UW0_S UW1_B UW1_B UW1_E EY_I EY_S EY_S EY0_B AO1_I AO1_S AO2_B EY_I EY_S EY0_B AY1_E AY1_E AY1_E AY1_I AY1_I AY1_S W_E W_I W_S OY_I OY_I OY_S OY_S OY0_B OY0_B UW_E UW_I UW_I UW_I UW_S SH_I SH_S W_B UW_E UW_E UW_E UW_E UW_E UW_E UW_I UW_I UW_I UW_S UW_S UW_S UW_S T_B T_E T_I T_I T_I T_I T_I T_I T_I UW0_S UW0_S UW0_S UW0_S UW0_S UW1_B UW1_B UW1_E UW1_E NG_S NG_S EY_B EY_E T_S N_B N_B N_E N_E N_E N_E UW0_S UW0_S UW1_B UW1_B UW1_B UW1_B UW1_E NG_B NG_B NG_B NG_E NG_I B_E B_E B_E B_E B_E B_E B_E B_E B_E B_I B_I B_S UW_E UW_I UW_S UW_S UW_S UW_S <eps> <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I SIL_I
1034-121119-0009 <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I K_E K_I K_S EY_I EY_S EY0_B UW2_E UW2_I UW2_S UW2_S EY_I EY_I EY_I EY_S EY_S EY0_B SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_B S_B S_E S_E OY0_E OY0_I OY0_I OY0_I OY0_S OY0_S UW0_B UW0_B UW0_E UW0_I AO1_I AO1_S AO2_B SH_I SH_S W_B Y_B Y_E Y_E Y_E Y_E Y_E Y_E Y_I NG_B NG_E NG_I EY_I EY_S EY0_B SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_E S_E S_E UW_E UW_E UW_I UW_S OY0_E OY0_I OY0_S OY0_S CH_B CH_E CH_E CH_E CH_E CH_E CH_E CH_E CH_E CH_E CH_E CH_I CH_I CH_I UW2_E UW2_E UW2_E UW2_E UW2_I UW2_I UW2_I UW2_S UW2_S UW2_S NG_S NG_S NG_S EY_B EY_E EY_E EY_E OY_I OY_I OY_S OY_S OY0_B OY0_B AO1_I AO1_S AO2_B SH_I SH_I SH_S W_B Y_B Y_B Y_E Y_I Y_I Y_I Y_I Y_I UW_E UW_I UW_I UW_S SH_I SH_S SH_S W_B W_B SIL_S SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_E SPN_I EY_I EY_I EY_S EY0_B EY0_E EY0_E EY0_E EY0_E EY0_E EY0_E EY0_E EY0_I EY0_I EY0_S EY0_S EY0_S EY0_S EY0_S AY2_S SH_B SH_B SH_E CH_B CH_B CH_B CH_B CH_E CH_E CH_E CH_I CH_I CH_I UW_E UW_I UW_S W_E W_E W_I W_I W_S W_S EY_I EY_I EY_S EY0_B EY0_B EY0_B UW0_B UW0_B UW0_B UW0_B UW0_B UW0_E UW0_E UW0_I UW_E UW_I UW_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_B S_E S_E S_E <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I
```
```
$ ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | utils/int2sym.pl -f 2- exp/mono/phones.txt | head -4
ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 SIL IH0_B F_E HH_B IY1_I D_E R_B AH1_I N_E AW1_B T_E AH0_B V_E T_B ER1_I N_I AH0_I P_E S_B IY1_I D_E HH_B IY1_E W_B UH1_I D_I AH0_I N_I T_E D_B R_I EH1_I S_E AH1_B P_E AH0_B N_I D_E T_B EY1_I K_E DH_B AH1_E B_B AH1_I G_I IY0_E T_B IH0_E G_B OW1_E F_B ER0_E M_B AO1_I R_E SIL
103-1241-0019 SIL AY1_S HH_B AE1_I D_E M_B EY1_I D_E AH1_B P_E M_B AY1_E M_B AY1_I N_I D_E SIL DH_B AH0_I T_E IH0_B F_E Y_B UW1_E D_B IH1_I D_I AH0_I N_E K_B AH1_I M_E F_B ER0_E M_B IY1_E T_B IH0_E N_B AY1_I T_E SIL
103-1241-0025 SIL IH1_B T_I S_E W_B ER1_I S_E DH_B AE1_I N_E SIL EH1_B N_I IY0_I TH_I IH2_I NG_E Y_B UW1_E K_B UH1_I D_E IH0_B M_I AE1_I JH_I AH0_I N_E SIL M_B IH1_I S_I IH0_I Z_E S_B P_I EH1_I N_I S_I ER0_E S_B EH1_I D_E IH0_B T_E W_B AH0_I Z_E W_B IH1_I K_I AH0_I D_E AH0_B V_E M_B IY1_E T_B IH0_E T_B AO1_I K_E L_B AY1_I K_E DH_B AE1_I T_E SIL
1034-121119-0009 SIL W_B AH0_I Z_E AH0_B S_I EH1_I N_I D_I IH0_I NG_E DH_B AH1_E S_B T_I EH1_I R_I Z_E L_B IY1_I D_I IH0_I NG_E T_B IH0_E SPN_S AH0_B P_I AA1_I R_I T_I M_I AH0_I N_I T_I S_E SIL
```
