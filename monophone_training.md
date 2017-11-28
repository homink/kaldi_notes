# What is found after the monophone training

When you need to understand complicatd systems, it is useful to analyze the output for drawing a bic picture. Here, we analyze the output of the monophone training in Kaldi. After running [steps/train_mono.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh), [utils/mkgraph.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/mkgraph.sh), and [steps/decode.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/decode.sh), you will find the followings in exp/mono folder.

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

When running [steps/train_mono.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/train_mono.sh), --nj was given by 20. This is stored in exp/mono/num_jobs file

```
$ more exp/mono/num_jobs
20
```

## What does fsts.x.gz have?

These are tables of FSTs where transcript is processed with a sequence of indexed number of words per utterance ID. 

```
$ fsts-to-transcripts 'ark:gunzip -c exp/mono/fsts.1.gz|' ark,t:- | head -4
fsts-to-transcripts 'ark:gunzip -c exp/mono/fsts.1.gz|' ark,t:-
103-1240-0015 85112 78821 152063 127347 124782 182467 157175 78818 197124 51026 186768 5411 173045 175861 23417 178313 71456 64424 116894
103-1241-0019 84746 76098 106437 186768 119160 114115 175810 85112 198712 47115 35043 64424 111012 178313 121877
103-1241-0025 89704 197011 175730 6520 198712 38672 85537 114954 165872 152997 89699 192633 194974 124782 111012 178313 173151 102060 175810
1034-121119-0009 192633 8588 175861 167203 100011 178313 3 6588

$ fsts-to-transcripts 'ark:gunzip -c exp/mono/fsts.1.gz|' ark,t:- | utils/int2sym.pl -f 2- data/lang/words.txt | head -4
fsts-to-transcripts 'ark:gunzip -c exp/mono/fsts.1.gz|' ark,t:-
LOG (fsts-to-transcripts[5.2.57~1391-97922]:main():fsts-to-transcripts.cc:97) Converted 128 FSTs, 0 with errors
103-1240-0015 IF HE'D RUN OUT OF TURNIP SEED HE WOULDN'T DRESS UP AND TAKE THE BUGGY TO GO FOR MORE
103-1241-0019 I HAD MADE UP MY MIND THAT IF YOU DIDN'T COME FOR ME TO NIGHT
103-1241-0025 IT'S WORSE THAN ANYTHING YOU COULD IMAGINE MISSUS SPENCER SAID IT WAS WICKED OF ME TO TALK LIKE THAT
1034-121119-0009 WAS ASCENDING THE STAIRS LEADING TO <UNK> APARTMENTS

$ grep -w "84746\|76098\|106437\|186768" data/lang/words.txt
HAD 76098
I 84746
MADE 106437
UP 186768

$ grep "103-1241-0019" data/train_2kshort/text
103-1241-0019 I HAD MADE UP MY MIND THAT IF YOU DIDN'T COME FOR ME TO NIGHT

$ grep "1034-121119-0009" data/train_2kshort/text
1034-121119-0009 WAS ASCENDING THE STAIRS LEADING TO DEBRAY'S APARTMENTS
```

## What does exp/mono/final.mdl has?

This is the HMM constructed with transition states & IDs, PDFs monophones.

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

## What does ali.x.gz work with exp/mono/final.mdl?
```
ali-to-pdf exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | utils/int2sym.pl -f 2- exp/mono/phones.txt | head -1
ali-to-pdf exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> <eps> SIL_E SIL_E SIL_E SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SIL_I SH_I SH_I SH_S W_B AY0_I AY0_I AY0_I AY0_S AY1_B EY2_I EY2_S B_B OY_I OY_I OY_S OY_S OY_S OY0_B OY0_B OY0_B OY0_B OY0_B AO1_I AO1_S AO2_B CH_B CH_B CH_E CH_E CH_I CH_I EY_I EY_I EY_I EY_I EY_S EY0_B UW0_B UW0_E UW0_E UW0_I UW0_I EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY1_S EY2_B EY2_B EY2_E UW_E UW_I UW_S EY_I EY_S EY0_B AY1_E AY1_I AY1_S AY1_S AY1_S AY1_S UW_E UW_E UW_E UW_I UW_I UW_I UW_I UW_S UW_S UW_S UW_S AY2_B AY2_B AY2_E AY2_E AY2_E AY2_E AY2_I UW0_B UW0_E UW0_I UW0_I UW0_I EY_I EY_S EY_S EY0_B EY0_B EY0_E EY0_E EY0_I EY0_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_B S_E OY_I OY_I OY_I OY_S OY_S OY_S OY_S OY_S OY_S OY_S OY0_B AO1_I AO1_S AO2_B EY2_I EY2_I EY2_S EY2_S B_B OY_I OY_I OY_S OY_S OY0_B OY0_B OY0_B K_E K_E K_I K_I K_I K_I K_S K_S AY_S AY0_B AY0_E AO1_I AO1_S AO2_B AO2_B EY_I EY_S EY0_B UW0_B UW0_B UW0_B UW0_E UW0_I UW_E UW_I UW_S AO1_I AO1_S AO2_B AO2_B CH_B CH_E CH_E CH_I OY0_E OY0_E OY0_I OY0_S OY0_S OY0_S SPN_S SPN_S SPN_S SPN_S SPN_S S_B S_B S_E S_E S_E EY_I EY_I EY_I EY_I EY_S EY0_B EY0_B EY0_E EY0_E EY0_I EY0_I EY0_I EY0_S EY_I EY_I EY_I EY_S EY0_B UW0_B UW0_E UW0_I UW0_I AO1_I AO1_S AO2_B UW_E UW_I UW_I UW_I UW_S Y_S Y_S Z_B Z_B Z_B Z_E Z_E UW0_S UW0_S UW0_S UW1_B UW1_B UW1_E UW1_E NG_B NG_B NG_B NG_E NG_I EY_I EY_S EY0_B Z_I Z_I Z_S Z_S Z_S Z_S Z_S AO_B AO_B AO_B EY_I EY_I EY_I EY_I EY_S EY_S EY0_B EY0_B AY_B AY_B AY_E AY_I AY_I AY_I OY_I OY_S OY_S OY_S OY_S OY0_B OY0_B OY0_B UW_E UW_E UW_E UW_I UW_I UW_S SH_I SH_S W_B W_B AY_B AY_E AY_E AY_I AY_I AY_I AY_I EY1_B EY1_B EY1_B EY1_B EY1_B EY1_E EY1_E EY1_I EY1_I AY0_I AY0_I AY0_I AY0_I AY0_I AY0_I AY0_S AY0_S AY0_S AY1_B AY2_B AY2_E AY2_I W_E W_E W_E W_E W_I W_I W_I W_S T_B T_E T_E T_E T_E T_I T_I T_I T_I T_I T_I T_I CH_B CH_E CH_I CH_I CH_I CH_I CH_I CH_I CH_I <eps> <eps> <eps> <eps> <eps> <eps> SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_E SIL_I

$ ali-to-post 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | head -1
ali-to-post 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 [ 4 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 16 1 ] [ 15 1 ] [ 15 1 ] [ 18 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 1382 1 ] [ 1381 1 ] [ 1384 1 ] [ 1386 1 ] [ 1124 1 ] [ 1123 1 ] [ 1123 1 ] [ 1126 1 ] [ 1128 1 ] [ 1838 1 ] [ 1840 1 ] [ 1842 1 ] [ 2066 1 ] [ 2065 1 ] [ 2068 1 ] [ 2067 1 ] [ 2067 1 ] [ 2070 1 ] [ 2069 1 ] [ 2069 1 ] [ 2069 1 ] [ 2069 1 ] [ 956 1 ] [ 958 1 ] [ 960 1 ] [ 1958 1 ] [ 1957 1 ] [ 1960 1 ] [ 1959 1 ] [ 1962 1 ] [ 1961 1 ] [ 1586 1 ] [ 1585 1 ] [ 1585 1 ] [ 1585 1 ] [ 1588 1 ] [ 1590 1 ] [ 332 1 ] [ 334 1 ] [ 333 1 ] [ 336 1 ] [ 335 1 ] [ 1790 1 ] [ 1789 1 ] [ 1789 1 ] [ 1789 1 ] [ 1789 1 ] [ 1789 1 ] [ 1789 1 ] [ 1789 1 ] [ 1792 1 ] [ 1791 1 ] [ 1794 1 ] [ 308 1 ] [ 310 1 ] [ 312 1 ] [ 1550 1 ] [ 1552 1 ] [ 1554 1 ] [ 1148 1 ] [ 1150 1 ] [ 1152 1 ] [ 1151 1 ] [ 1151 1 ] [ 1151 1 ] [ 302 1 ] [ 301 1 ] [ 301 1 ] [ 304 1 ] [ 303 1 ] [ 303 1 ] [ 303 1 ] [ 306 1 ] [ 305 1 ] [ 305 1 ] [ 305 1 ] [ 1226 1 ] [ 1225 1 ] [ 1228 1 ] [ 1227 1 ] [ 1227 1 ] [ 1227 1 ] [ 1230 1 ] [ 338 1 ] [ 340 1 ] [ 342 1 ] [ 341 1 ] [ 341 1 ] [ 1562 1 ] [ 1564 1 ] [ 1563 1 ] [ 1566 1 ] [ 1565 1 ] [ 1628 1 ] [ 1627 1 ] [ 1630 1 ] [ 1632 1 ] [ 182 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 181 1 ] [ 184 1 ] [ 183 1 ] [ 183 1 ] [ 186 1 ] [ 2066 1 ] [ 2065 1 ] [ 2065 1 ] [ 2068 1 ] [ 2067 1 ] [ 2067 1 ] [ 2067 1 ] [ 2067 1 ] [ 2067 1 ] [ 2067 1 ] [ 2070 1 ] [ 956 1 ] [ 958 1 ] [ 960 1 ] [ 1838 1 ] [ 1837 1 ] [ 1840 1 ] [ 1839 1 ] [ 1842 1 ] [ 2060 1 ] [ 2059 1 ] [ 2062 1 ] [ 2061 1 ] [ 2064 1 ] [ 2063 1 ] [ 2063 1 ] [ 638 1 ] [ 637 1 ] [ 640 1 ] [ 639 1 ] [ 639 1 ] [ 639 1 ] [ 642 1 ] [ 641 1 ] [ 1082 1 ] [ 1084 1 ] [ 1086 1 ] [ 962 1 ] [ 964 1 ] [ 966 1 ] [ 965 1 ] [ 1562 1 ] [ 1564 1 ] [ 1566 1 ] [ 338 1 ] [ 337 1 ] [ 337 1 ] [ 340 1 ] [ 342 1 ] [ 308 1 ] [ 310 1 ] [ 312 1 ] [ 950 1 ] [ 952 1 ] [ 954 1 ] [ 953 1 ] [ 1970 1 ] [ 1972 1 ] [ 1971 1 ] [ 1974 1 ] [ 2162 1 ] [ 2161 1 ] [ 2164 1 ] [ 2166 1 ] [ 2165 1 ] [ 2165 1 ] [ 188 1 ] [ 187 1 ] [ 187 1 ] [ 187 1 ] [ 187 1 ] [ 190 1 ] [ 189 1 ] [ 192 1 ] [ 191 1 ] [ 191 1 ] [ 1574 1 ] [ 1573 1 ] [ 1573 1 ] [ 1573 1 ] [ 1576 1 ] [ 1578 1 ] [ 1577 1 ] [ 1628 1 ] [ 1627 1 ] [ 1630 1 ] [ 1629 1 ] [ 1629 1 ] [ 1632 1 ] [ 1550 1 ] [ 1549 1 ] [ 1549 1 ] [ 1552 1 ] [ 1554 1 ] [ 338 1 ] [ 340 1 ] [ 342 1 ] [ 341 1 ] [ 956 1 ] [ 958 1 ] [ 960 1 ] [ 302 1 ] [ 304 1 ] [ 303 1 ] [ 303 1 ] [ 306 1 ] [ 746 1 ] [ 745 1 ] [ 748 1 ] [ 747 1 ] [ 747 1 ] [ 750 1 ] [ 749 1 ] [ 356 1 ] [ 355 1 ] [ 355 1 ] [ 358 1 ] [ 357 1 ] [ 360 1 ] [ 359 1 ] [ 1478 1 ] [ 1477 1 ] [ 1477 1 ] [ 1480 1 ] [ 1482 1 ] [ 1580 1 ] [ 1582 1 ] [ 1584 1 ] [ 782 1 ] [ 781 1 ] [ 784 1 ] [ 783 1 ] [ 783 1 ] [ 783 1 ] [ 783 1 ] [ 786 1 ] [ 785 1 ] [ 785 1 ] [ 1586 1 ] [ 1585 1 ] [ 1585 1 ] [ 1585 1 ] [ 1588 1 ] [ 1587 1 ] [ 1590 1 ] [ 1589 1 ] [ 1010 1 ] [ 1009 1 ] [ 1012 1 ] [ 1014 1 ] [ 1013 1 ] [ 1013 1 ] [ 2036 1 ] [ 2038 1 ] [ 2037 1 ] [ 2037 1 ] [ 2037 1 ] [ 2040 1 ] [ 2039 1 ] [ 2039 1 ] [ 302 1 ] [ 301 1 ] [ 301 1 ] [ 304 1 ] [ 303 1 ] [ 306 1 ] [ 1388 1 ] [ 1390 1 ] [ 1392 1 ] [ 1391 1 ] [ 998 1 ] [ 1000 1 ] [ 999 1 ] [ 1002 1 ] [ 1001 1 ] [ 1001 1 ] [ 1001 1 ] [ 1700 1 ] [ 1699 1 ] [ 1699 1 ] [ 1699 1 ] [ 1699 1 ] [ 1702 1 ] [ 1701 1 ] [ 1704 1 ] [ 1703 1 ] [ 1118 1 ] [ 1117 1 ] [ 1117 1 ] [ 1117 1 ] [ 1117 1 ] [ 1117 1 ] [ 1120 1 ] [ 1119 1 ] [ 1119 1 ] [ 1122 1 ] [ 1196 1 ] [ 1198 1 ] [ 1200 1 ] [ 1454 1 ] [ 1453 1 ] [ 1453 1 ] [ 1453 1 ] [ 1456 1 ] [ 1455 1 ] [ 1455 1 ] [ 1458 1 ] [ 482 1 ] [ 484 1 ] [ 483 1 ] [ 483 1 ] [ 483 1 ] [ 486 1 ] [ 485 1 ] [ 485 1 ] [ 485 1 ] [ 485 1 ] [ 485 1 ] [ 485 1 ] [ 1964 1 ] [ 1966 1 ] [ 1968 1 ] [ 1967 1 ] [ 1967 1 ] [ 1967 1 ] [ 1967 1 ] [ 1967 1 ] [ 1967 1 ] [ 4 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 16 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 15 1 ] [ 18 1 ]

$ ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:- | utils/int2sym.pl -f 2- exp/mono/phones.txt | head -1
ali-to-phones exp/mono/final.mdl 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:-
103-1240-0015 SIL IH0_B F_E HH_B IY1_I D_E R_B AH1_I N_E AW1_B T_E AH0_B V_E T_B ER1_I N_I AH0_I P_E S_B IY1_I D_E HH_B IY1_E W_B UH1_I D_I AH0_I N_I T_E D_B R_I EH1_I S_E AH1_B P_E AH0_B N_I D_E T_B EY1_I K_E DH_B AH1_E B_B AH1_I G_I IY0_E T_B IH0_E G_B OW1_E F_B ER0_E M_B AO1_I R_E SIL

$ show-alignments data/lang/phones.txt exp/mono/0.mdl "ark:gunzip -c exp/mono/ali.1.gz |" | head -2
show-alignments data/lang/phones.txt exp/mono/0.mdl 'ark:gunzip -c exp/mono/ali.1.gz |'
103-1240-0015  [ 4 1 1 1 1 1 1 1 1 16 15 15 18 17 17 17 17 17 17 17 ] [ 1382 1381 1384 1386 ] [ 1124 1123 1123 1126 1128 ] [ 1838 1840 1842 ] [ 2066 2065 2068 2067 2067 2070 2069 2069 2069 2069 ] [ 956 958 960 ] [ 1958 1957 1960 1959 1962 1961 ] [ 1586 1585 1585 1585 1588 1590 ] [ 332 334 333 336 335 ] [ 1790 1789 1789 1789 1789 1789 1789 1789 1792 1791 1794 ] [ 308 310 312 ] [ 1550 1552 1554 ] [ 1148 1150 1152 1151 1151 1151 ] [ 302 301 301 304 303 303 303 306 305 305 305 ] [ 1226 1225 1228 1227 1227 1227 1230 ] [ 338 340 342 341 341 ] [ 1562 1564 1563 1566 1565 ] [ 1628 1627 1630 1632 ] [ 182 181 181 181 181 181 181 181 181 184 183 183 186 ] [ 2066 2065 2065 2068 2067 2067 2067 2067 2067 2067 2070 ] [ 956 958 960 ] [ 1838 1837 1840 1839 1842 ] [ 2060 2059 2062 2061 2064 2063 2063 ] [ 638 637 640 639 639 639 642 641 ] [ 1082 1084 1086 ] [ 962 964 966 965 ] [ 1562 1564 1566 ] [ 338 337 337 340 342 ] [ 308 310 312 ] [ 950 952 954 953 ] [ 1970 1972 1971 1974 ] [ 2162 2161 2164 2166 2165 2165 ] [ 188 187 187 187 187 190 189 192 191 191 ] [ 1574 1573 1573 1573 1576 1578 1577 ] [ 1628 1627 1630 1629 1629 1632 ] [ 1550 1549 1549 1552 1554 ] [ 338 340 342 341 ] [ 956 958 960 ] [ 302 304 303 303 306 ] [ 746 745 748 747 747 750 749 ] [ 356 355 355 358 357 360 359 ] [ 1478 1477 1477 1480 1482 ] [ 1580 1582 1584 ] [ 782 781 784 783 783 783 783 786 785 785 ] [ 1586 1585 1585 1585 1588 1587 1590 1589 ] [ 1010 1009 1012 1014 1013 1013 ] [ 2036 2038 2037 2037 2037 2040 2039 2039 ] [ 302 301 301 304 303 306 ] [ 1388 1390 1392 1391 ] [ 998 1000 999 1002 1001 1001 1001 ] [ 1700 1699 1699 1699 1699 1702 1701 1704 1703 ] [ 1118 1117 1117 1117 1117 1117 1120 1119 1119 1122 ] [ 1196 1198 1200 ] [ 1454 1453 1453 1453 1456 1455 1455 1458 ] [ 482 484 483 483 483 486 485 485 485 485 485 485 ] [ 1964 1966 1968 1967 1967 1967 1967 1967 1967 ] [ 4 1 1 1 1 1 16 15 15 15 15 15 15 15 15 15 15 15 15 15 18 ]
103-1240-0015  SIL                                                    IH0_B                   F_E                          HH_B               IY1_I                                                 D_E             R_B                               AH1_I                             N_E                     AW1_B                                                      T_E             AH0_B              V_E                               T_B                                             ER1_I                                  N_I                     AH0_I                        P_E                     S_B                                                     IY1_I                                                      D_E             HH_B                         IY1_E                                  W_B                                 UH1_I              D_I                 AH0_I              N_I                     T_E             D_B                 R_I                     EH1_I                             S_E                                         AH1_B                                  P_E                               AH0_B                        N_I                 D_E             T_B                     EY1_I                           K_E                             DH_B                         AH1_E              B_B                                         AH1_I                                       G_I                               IY0_E                                       T_B                         IH0_E                   G_B                                  OW1_E                                            F_B                                                   ER0_E              M_B                                         AO1_I                                               R_E                                              SIL

$ ali-to-post 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:exp/mono/post.1
ali-to-post 'ark:gunzip -c exp/mono/ali.1.gz|' ark,t:exp/mono/post.1
LOG (ali-to-post[5.2.57~1391-97922]:main():ali-to-post.cc:73) Converted 128 alignments.

$ post-to-weights ark:exp/mono/post.1 ark,t:- | head -1
post-to-weights ark:exp/mono/post.1 ark,t:-
103-1240-0015  [ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ]

$ post-to-pdf-post exp/mono/final.mdl ark:exp/mono/post.1 ark,t:- | head -1
post-to-pdf-post exp/mono/final.mdl ark:exp/mono/post.1 ark,t:-
103-1240-0015 [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 4 1 ] [ 85 1 ] [ 85 1 ] [ 86 1 ] [ 87 1 ] [ 73 1 ] [ 73 1 ] [ 73 1 ] [ 74 1 ] [ 75 1 ] [ 109 1 ] [ 110 1 ] [ 111 1 ] [ 121 1 ] [ 121 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 123 1 ] [ 123 1 ] [ 123 1 ] [ 123 1 ] [ 123 1 ] [ 61 1 ] [ 62 1 ] [ 63 1 ] [ 115 1 ] [ 115 1 ] [ 116 1 ] [ 116 1 ] [ 117 1 ] [ 117 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 19 1 ] [ 20 1 ] [ 20 1 ] [ 21 1 ] [ 21 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 106 1 ] [ 107 1 ] [ 107 1 ] [ 108 1 ] [ 16 1 ] [ 17 1 ] [ 18 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 76 1 ] [ 77 1 ] [ 78 1 ] [ 78 1 ] [ 78 1 ] [ 78 1 ] [ 16 1 ] [ 16 1 ] [ 16 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 18 1 ] [ 18 1 ] [ 18 1 ] [ 18 1 ] [ 79 1 ] [ 79 1 ] [ 80 1 ] [ 80 1 ] [ 80 1 ] [ 80 1 ] [ 81 1 ] [ 19 1 ] [ 20 1 ] [ 21 1 ] [ 21 1 ] [ 21 1 ] [ 97 1 ] [ 98 1 ] [ 98 1 ] [ 99 1 ] [ 99 1 ] [ 100 1 ] [ 100 1 ] [ 101 1 ] [ 102 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 12 1 ] [ 121 1 ] [ 121 1 ] [ 121 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 123 1 ] [ 61 1 ] [ 62 1 ] [ 63 1 ] [ 109 1 ] [ 109 1 ] [ 110 1 ] [ 110 1 ] [ 111 1 ] [ 121 1 ] [ 121 1 ] [ 122 1 ] [ 122 1 ] [ 123 1 ] [ 123 1 ] [ 123 1 ] [ 40 1 ] [ 40 1 ] [ 41 1 ] [ 41 1 ] [ 41 1 ] [ 41 1 ] [ 42 1 ] [ 42 1 ] [ 70 1 ] [ 71 1 ] [ 72 1 ] [ 61 1 ] [ 62 1 ] [ 63 1 ] [ 63 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 19 1 ] [ 19 1 ] [ 19 1 ] [ 20 1 ] [ 21 1 ] [ 16 1 ] [ 17 1 ] [ 18 1 ] [ 61 1 ] [ 62 1 ] [ 63 1 ] [ 63 1 ] [ 115 1 ] [ 116 1 ] [ 116 1 ] [ 117 1 ] [ 124 1 ] [ 124 1 ] [ 125 1 ] [ 126 1 ] [ 126 1 ] [ 126 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 10 1 ] [ 11 1 ] [ 11 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 99 1 ] [ 100 1 ] [ 100 1 ] [ 101 1 ] [ 101 1 ] [ 101 1 ] [ 102 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 19 1 ] [ 20 1 ] [ 21 1 ] [ 21 1 ] [ 61 1 ] [ 62 1 ] [ 63 1 ] [ 16 1 ] [ 17 1 ] [ 17 1 ] [ 17 1 ] [ 18 1 ] [ 46 1 ] [ 46 1 ] [ 47 1 ] [ 47 1 ] [ 47 1 ] [ 48 1 ] [ 48 1 ] [ 22 1 ] [ 22 1 ] [ 22 1 ] [ 23 1 ] [ 23 1 ] [ 24 1 ] [ 24 1 ] [ 91 1 ] [ 91 1 ] [ 91 1 ] [ 92 1 ] [ 93 1 ] [ 97 1 ] [ 98 1 ] [ 99 1 ] [ 49 1 ] [ 49 1 ] [ 50 1 ] [ 50 1 ] [ 50 1 ] [ 50 1 ] [ 50 1 ] [ 51 1 ] [ 51 1 ] [ 51 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 97 1 ] [ 98 1 ] [ 98 1 ] [ 99 1 ] [ 99 1 ] [ 67 1 ] [ 67 1 ] [ 68 1 ] [ 69 1 ] [ 69 1 ] [ 69 1 ] [ 121 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 122 1 ] [ 123 1 ] [ 123 1 ] [ 123 1 ] [ 16 1 ] [ 16 1 ] [ 16 1 ] [ 17 1 ] [ 17 1 ] [ 18 1 ] [ 85 1 ] [ 86 1 ] [ 87 1 ] [ 87 1 ] [ 67 1 ] [ 68 1 ] [ 68 1 ] [ 69 1 ] [ 69 1 ] [ 69 1 ] [ 69 1 ] [ 103 1 ] [ 103 1 ] [ 103 1 ] [ 103 1 ] [ 103 1 ] [ 104 1 ] [ 104 1 ] [ 105 1 ] [ 105 1 ] [ 73 1 ] [ 73 1 ] [ 73 1 ] [ 73 1 ] [ 73 1 ] [ 73 1 ] [ 74 1 ] [ 74 1 ] [ 74 1 ] [ 75 1 ] [ 79 1 ] [ 80 1 ] [ 81 1 ] [ 88 1 ] [ 88 1 ] [ 88 1 ] [ 88 1 ] [ 89 1 ] [ 89 1 ] [ 89 1 ] [ 90 1 ] [ 31 1 ] [ 32 1 ] [ 32 1 ] [ 32 1 ] [ 32 1 ] [ 33 1 ] [ 33 1 ] [ 33 1 ] [ 33 1 ] [ 33 1 ] [ 33 1 ] [ 33 1 ] [ 115 1 ] [ 116 1 ] [ 117 1 ] [ 117 1 ] [ 117 1 ] [ 117 1 ] [ 117 1 ] [ 117 1 ] [ 117 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 0 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 3 1 ] [ 4 1 ]

$ post-to-phone-post exp/mono/final.mdl ark:exp/mono/post.1 ark,t:- | head -1
post-to-phone-post exp/mono/final.mdl ark:exp/mono/post.1 ark,t:-
103-1240-0015 [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 211 1 ] [ 211 1 ] [ 211 1 ] [ 211 1 ] [ 168 1 ] [ 168 1 ] [ 168 1 ] [ 168 1 ] [ 168 1 ] [ 287 1 ] [ 287 1 ] [ 287 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 140 1 ] [ 140 1 ] [ 140 1 ] [ 307 1 ] [ 307 1 ] [ 307 1 ] [ 307 1 ] [ 307 1 ] [ 307 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 36 1 ] [ 36 1 ] [ 36 1 ] [ 36 1 ] [ 36 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 279 1 ] [ 32 1 ] [ 32 1 ] [ 32 1 ] [ 239 1 ] [ 239 1 ] [ 239 1 ] [ 172 1 ] [ 172 1 ] [ 172 1 ] [ 172 1 ] [ 172 1 ] [ 172 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 185 1 ] [ 185 1 ] [ 185 1 ] [ 185 1 ] [ 185 1 ] [ 185 1 ] [ 185 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 241 1 ] [ 241 1 ] [ 241 1 ] [ 241 1 ] [ 241 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 11 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 325 1 ] [ 140 1 ] [ 140 1 ] [ 140 1 ] [ 287 1 ] [ 287 1 ] [ 287 1 ] [ 287 1 ] [ 287 1 ] [ 324 1 ] [ 324 1 ] [ 324 1 ] [ 324 1 ] [ 324 1 ] [ 324 1 ] [ 324 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 87 1 ] [ 161 1 ] [ 161 1 ] [ 161 1 ] [ 141 1 ] [ 141 1 ] [ 141 1 ] [ 141 1 ] [ 241 1 ] [ 241 1 ] [ 241 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 32 1 ] [ 32 1 ] [ 32 1 ] [ 139 1 ] [ 139 1 ] [ 139 1 ] [ 139 1 ] [ 309 1 ] [ 309 1 ] [ 309 1 ] [ 309 1 ] [ 341 1 ] [ 341 1 ] [ 341 1 ] [ 341 1 ] [ 341 1 ] [ 341 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 12 1 ] [ 243 1 ] [ 243 1 ] [ 243 1 ] [ 243 1 ] [ 243 1 ] [ 243 1 ] [ 243 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 252 1 ] [ 239 1 ] [ 239 1 ] [ 239 1 ] [ 239 1 ] [ 239 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 37 1 ] [ 140 1 ] [ 140 1 ] [ 140 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 105 1 ] [ 105 1 ] [ 105 1 ] [ 105 1 ] [ 105 1 ] [ 105 1 ] [ 105 1 ] [ 40 1 ] [ 40 1 ] [ 40 1 ] [ 40 1 ] [ 40 1 ] [ 40 1 ] [ 40 1 ] [ 227 1 ] [ 227 1 ] [ 227 1 ] [ 227 1 ] [ 227 1 ] [ 244 1 ] [ 244 1 ] [ 244 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 111 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 245 1 ] [ 149 1 ] [ 149 1 ] [ 149 1 ] [ 149 1 ] [ 149 1 ] [ 149 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 320 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 31 1 ] [ 212 1 ] [ 212 1 ] [ 212 1 ] [ 212 1 ] [ 147 1 ] [ 147 1 ] [ 147 1 ] [ 147 1 ] [ 147 1 ] [ 147 1 ] [ 147 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 264 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 167 1 ] [ 180 1 ] [ 180 1 ] [ 180 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 223 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 61 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 308 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ] [ 1 1 ]
```

## What dose tree file has?
```
$ copy-tree --binary=false exp/mono/tree - | head -n 17
copy-tree --binary=false exp/mono/tree -
ContextDependency 1 0 ToPdf SE 0 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 ]
{ SE 0 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 ]
{ SE 0 [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ]
{ SE 0 [ 1 2 3 4 5 6 7 8 9 10 ]
{ SE 0 [ 1 2 3 4 5 ]
{ TE -1 5 ( CE 0 CE 1 CE 2 CE 3 CE 4 )
TE -1 5 ( CE 5 CE 6 CE 7 CE 8 CE 9 )
}
SE 0 [ 11 12 13 14 ]
{ TE -1 3 ( CE 10 CE 11 CE 12 )
SE 0 [ 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 ]
{ TE -1 3 ( CE 13 CE 14 CE 15 )
TE -1 3 ( CE 16 CE 17 CE 18 )
}
}
}
SE 0 [ 35 36 37 38 39 40 41 42 ]
LOG (copy-tree[5.2.57~1391-97922]:main():copy-tree.cc:55) Copied tree
```

## [steps/decode.sh](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/decode.sh) produces the lattice and scored information

The lattice (i.e. [CompactLattice type known in FSA, acceptor, format, with an arc format](http://codingandlearning.blogspot.com/2014/01/kaldi-lattices.html)) is formed as follows:

[start state id] [end state id] [input & output symbol] [weight]

The input/output symbol are usually the word ids and the weight is:

[the graph cost],[the acoustic cost],[a string sequence of integers]

The integers represent the transition ids, i.e. the frame-alignment for this word symbol.


```
>> lattice-copy "ark:gunzip -c exp/mono/decode_nosp_tgsmall_test_clean/lat.1.gz|" ark,t:- | utils/int2sym.pl -f 3 data/lang_nosp/words.txt | head -10
lattice-copy 'ark:gunzip -c exp/mono/decode_nosp_tgsmall_test_clean/lat.1.gz|' ark,t:-
1089-134686-0000
0 1 HE 9.12696,5837.53,4_1_1_1_1_1_1_16_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_18_17_17_17_17_17_1838_1840_1839_1842_2060_2059_2059_2059_2062_2061_2064
0 196 THE 17.022,8215.73,4_1_1_1_1_1_1_16_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_18_17_17_17_17_1478_1480_1482_1481_2036_2035_2035_2035_2035_2038_2037_2040_2039_2039_2039_1838_1837_1840_1842_1706_1705_1705_1705_1705_1705_1708_1707_1707_1707_1707_1710_1634_1633_1633_1633_1636_1635_1638
0 212 THEY 16.0285,7389.03,4_1_1_1_1_1_1_16_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_18_17_17_17_17_1478_1480_1482_1481_740_739_739_742_744_743_743_743_743_743_1838_1837_1837_1840_1842_1706_1705_1705_1705_1705_1705_1708_1707_1707_1707
0 302 SHE 17.0534,8358.78,4_1_1_1_1_1_1_16_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_15_18_17_17_17_17_17_614_616_618_2060_2059_2059_2059_2059_2062_2061_2064_2063_2063_2063_1838_1837_1840_1842_1706_1705_1705_1705_1705_1705_1708_1707_1707_1707_1707_1710_1634_1633_1633_1633_1636_1635_1638_1637
1 2 HOPED 9.31398,3645.34,2063_2063_2063_1838_1837_1840_1842_1706_1705_1705_1705_1705_1705_1708_1707_1707_1707_1707_1710_1634_1633_1633_1633_1636_1635_1638_1637_308_307_307_310
2 3 THERE 5.06083,843.538,
2 60 THEY 3.9055,600.173,309_312_1478_1477_1477_1477_1480_1482_740_739_742_744
2 186 IT 4.3805,2018.23,309_309_312_311_311_311_311_311_1382_1381_1384_1386_308_310_312_638_637_637_637_637_637_640_639_639_642
3 4 WOULD 1.94562,1761.93,309_312_1478_1477_1477_1477_1480_1482_2162_2161_2164_2166_1964_1966_1968_638_637_637_637_637_637_640_639_639_642_1082_1084

$ head -4 exp/mono/decode_nosp_tgsmall_test_clean/scoring/test_filt.txt
1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE
1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
1089-134686-0002 AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
1089-134686-0003 HELLO BERTIE ANY GOOD IN YOUR MIND

$ more exp/mono/decode_nosp_tgsmall_test_clean/wer_7_0.0
compute-wer --text --mode=present ark:exp/mono/decode_nosp_tgsmall_test_clean/scoring/test_filt.txt ark,p:-

%WER 43.73 [ 22992 / 52576, 1267 ins, 5064 del, 16661 sub ]
%SER 95.50 [ 2502 / 2620 ]
Scored 2620 sentences, 0 not present in hyp.
```
