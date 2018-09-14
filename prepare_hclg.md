# How to prepare HCLG.fst

Kaldi requires HCLG.fst for the WFST decoding. More details are found in [here](http://kaldi-asr.org/doc/graph.html). We review here step by step with the wsj recipe.

## Making L.fst

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/run.sh#L45
```
local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;

ls data/local/dict_nosp/
cmudict              lexicon1_raw_nosil.txt  lexiconp.txt  nonsilence_phones.txt  silence_phones.txt
extra_questions.txt  lexicon2_raw.txt        lexicon.txt   optional_silence.txt

head -3 data/local/dict_nosp/*.txt
==> data/local/dict_nosp/extra_questions.txt <==
SIL SPN NSN 
EY0 AY0 AW0 ER0 AH0 OY0 IY0 IH0 EH0 UH0 OW0 UW0 AO0 AE0 AA0 
EY1 AY1 AW1 ER1 AH1 OY1 IY1 IH1 EH1 UH1 OW1 UW1 AO1 AE1 AA1 

==> data/local/dict_nosp/lexicon1_raw_nosil.txt <==
!EXCLAMATION-POINT  EH2 K S K L AH0 M EY1 SH AH0 N P OY2 N T
"CLOSE-QUOTE  K L OW1 Z K W OW1 T
"DOUBLE-QUOTE  D AH1 B AH0 L K W OW1 T

==> data/local/dict_nosp/lexicon2_raw.txt <==
A42128  EY1 F AO1 R T UW1 W AH1 N T UW1 EY1 T
AAA  T R IH2 P AH0 L EY1
AABERG  AA1 B ER0 G

==> data/local/dict_nosp/lexiconp.txt <==
A42128  1.0     EY1 F AO1 R T UW1 W AH1 N T UW1 EY1 T
AAA  1.0        T R IH2 P AH0 L EY1
AABERG  1.0     AA1 B ER0 G

==> data/local/dict_nosp/lexicon.txt <==
A42128  EY1 F AO1 R T UW1 W AH1 N T UW1 EY1 T
AAA  T R IH2 P AH0 L EY1
AABERG  AA1 B ER0 G

==> data/local/dict_nosp/nonsilence_phones.txt <==
JH 
Y 
TH 

==> data/local/dict_nosp/optional_silence.txt <==
SIL

==> data/local/dict_nosp/silence_phones.txt <==
SIL
SPN
NSN
```

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/run.sh#L47
```
utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;
                        
ls data/local/lang_tmp_nosp/
align_lexicon.txt  lexiconp_disambig.txt  lexiconp.txt  lex_ndisambig  phone_map.txt

head -3 data/local/lang_tmp_nosp/*
==> data/local/lang_tmp_nosp/align_lexicon.txt <==
A42128 EY1_B F_I AO1_I R_I T_I UW1_I W_I AH1_I N_I T_I UW1_I EY1_I T_E
AAA T_B R_I IH2_I P_I AH0_I L_I EY1_E
AABERG AA1_B B_I ER0_I G_E

==> data/local/lang_tmp_nosp/lexiconp_disambig.txt <==
A42128  1.0     EY1_B F_I AO1_I R_I T_I UW1_I W_I AH1_I N_I T_I UW1_I EY1_I T_E
AAA     1.0     T_B R_I IH2_I P_I AH0_I L_I EY1_E
AABERG  1.0     AA1_B B_I ER0_I G_E

==> data/local/lang_tmp_nosp/lexiconp.txt <==
A42128 1.0 EY1_B F_I AO1_I R_I T_I UW1_I W_I AH1_I N_I T_I UW1_I EY1_I T_E
AAA 1.0 T_B R_I IH2_I P_I AH0_I L_I EY1_E
AABERG 1.0 AA1_B B_I ER0_I G_E

==> data/local/lang_tmp_nosp/lex_ndisambig <==
14

==> data/local/lang_tmp_nosp/phone_map.txt <==
SIL SIL SIL_B SIL_E SIL_I SIL_S 
SPN SPN SPN_B SPN_E SPN_I SPN_S 
NSN NSN NSN_B NSN_E NSN_I NSN_S 

ls data/lang_nosp/
L_disambig.fst  L_disambig.fst.sym  L_disambig.fst.txt  L.fst  L.fst.sym  L.fst.txt  oov.int  oov.txt  phones  phones.txt  topo  words.txt

ls data/lang_nosp/phones
align_lexicon.int  context_indep.txt  extra_questions.int  nonsilence.txt        roots.int  silence.csl           wdisambig.txt
align_lexicon.txt  disambig.csl       extra_questions.txt  optional_silence.csl  roots.txt  silence.int           wdisambig_words.int
context_indep.csl  disambig.int       nonsilence.csl       optional_silence.int  sets.int   silence.txt           word_boundary.int
context_indep.int  disambig.txt       nonsilence.int       optional_silence.txt  sets.txt   wdisambig_phones.int  word_boundary.txt
```

The prepared above files are used to make L.fst and L_disambig.fst. These fst files inlcude the following information. It is interesting to see that input symbol is the first phone of the word in the lexicon and output symbol is the correspoing word such as 'EH1_B   "END-QUOTE' in L.fst and L_disambig.fst.

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/utils/prepare_lang.sh#L460
```
utils/lang/make_lexicon_fst.py $grammar_opts --sil-prob=$sil_prob --sil-phone=$silphone \
    $tmpdir/lexiconp.txt | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
        --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L.fst | head
0       1       <eps>   <eps>   0.693147182
0       1       SIL     <eps>   0.693147182
1       217485  EH2_B   !EXCLAMATION-POINT
1       1       SIL_S   !SIL    0.693147182
1       2       SIL_S   !SIL    0.693147182
1       116624  K_B     "CLOSE-QUOTE
1       187220  D_B     "DOUBLE-QUOTE
1       205790  EH1_B   "END-OF-QUOTE
1       206073  EH1_B   "END-QUOTE
1       323251  IH1_B   "IN-QUOTES
```
https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/utils/prepare_lang.sh#L538
```
utils/lang/make_lexicon_fst.py $grammar_opts \
        --sil-prob=$sil_prob --sil-phone=$silphone --sil-disambig='#'$ndisambig \
        $tmpdir/lexiconp_disambig.txt | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
         --keep_isymbols=false --keep_osymbols=false |   \
    fstaddselfloops  $dir/phones/wdisambig_phones.int $dir/phones/wdisambig_words.int | \
    fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L_disambig.fst | head
0       1       <eps>   <eps>   0.693147182
0       2       SIL     <eps>   0.693147182
1       226184  EH2_B   !EXCLAMATION-POINT
1       3       SIL_S   !SIL    0.693147182
1       1       SIL_S   !SIL    0.693147182
1       121743  K_B     "CLOSE-QUOTE
1       194797  D_B     "DOUBLE-QUOTE
1       214216  EH1_B   "END-OF-QUOTE
1       214503  EH1_B   "END-QUOTE
1       336665  IH1_B   "IN-QUOTES

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L_disambig.fst | grep "#" data/lang_nosp/L_disambig.fst.sym | head
1       612518  SH_B    #SHARP-SIGN
1       1       #0      #0
2       1       #14     <eps>
34      1       #1      <eps>   0.693147182
34      3       #1      <eps>   0.693147182
46      1       #1      <eps>   0.693147182
46      3       #1      <eps>   0.693147182
61      1       #1      <eps>   0.693147182
61      3       #1      <eps>   0.693147182
66      1       #1      <eps>   0.693147182

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp/L.fst | grep "#" data/lang_nosp/L.fst.sym | head
1       586727  SH_B    #SHARP-SIGN
```

## Making G.fst

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/run.sh#L50
```
local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;

ls data/lang_nosp_test_tg_5k/
G.fst  L_disambig.fst  L_disambig.fst.sym  L_disambig.fst.txt  L.fst  L.fst.sym  L.fst.txt  oov.int  oov.txt  phones  phones.txt  topo  words.txt
```

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/local/wsj_format_data.sh#L51
```
  gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst
```

G.fst includs the following information. It is interesting to see input and output symbols are word ids.

```
fstprint --isymbols=data/lang_nosp_test_tg_5k/words.txt --osymbols=data/lang_nosp_test_tg_5k/words.txt data/lang_nosp_test_tg_5k/G.fst | head
2       8050    <UNK>   <UNK>   2.9495194
2       8051    A       A       4.42639732
2       8052    A.      A.      7.31982613
2       7       ABANDONED       ABANDONED       13.2024698
2       9       ABLE    ABLE    14.0942612
2       8053    ABOUT   ABOUT   7.18028927
2       8054    ABOVE   ABOVE   10.2153263
2       8055    ABROAD  ABROAD  12.2217989
2       14      ABRUPT  ABRUPT  13.7702875
2       15      ABSENCE ABSENCE 15.496767

fstprint --isymbols=data/lang_nosp_test_tg_5k/words.txt --osymbols=data/lang_nosp_test_tg_5k/words.txt data/lang_nosp_test_tg_5k/G.fst | tail
170444  33830   STOCKHOLM       STOCKHOLM       2.11031914
170444  590     #0      <eps>   3.07913184
170445  134423  PRICES  PRICES  0.133549929
170445  3987    #0      <eps>   2.04869056
170446  140651  EXCHANGE        EXCHANGE        0.251212031
170446  4257    #0      <eps>   1.26320744
170447  140809  AND     AND     0.0741432458
170447  4259    #0      <eps>   2.25016212
170448  143076  BASED   BASED   1.473194
170448  4391    #0      <eps>   0.226500243
```

## Making LG.fst

https://github.com/kaldi-asr/kaldi/blob/0cf2e23a7d8610e18b7f55ce4e09771b92989477/egs/wsj/s5/utils/mkgraph.sh#L100
```
fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial > $lang/tmp/LG.fst.$$ || exit 1;
    
fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp_test_tgpr/tmp/LG.fst | head
0       1       SIL     <eps>   0.698661923
0       2       SPN_S   <UNK>   4.45802927
0       3       JH_B    <eps>   5.54836178
0       4       Y_B     <eps>   5.29441261
0       5       TH_B    <eps>   6.90070009
0       6       EY0_B   <eps>   11.958147
0       7       EY1_B   <eps>   7.98114967
0       8       EY1_S   <eps>   4.89023113
0       9       EY2_B   AVIATION        13.7248697
0       10      CH_B    <eps>   7.19705915

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp_test_tgpr/tmp/LG.fst | tail
2122933 62      AE2_B   <eps>   8.24591923
2122933 63      F_B     <eps>   3.31286383
2122933 64      L_B     <eps>   4.357862
2122933 65      T_B     <eps>   3.90932679
2122933 66      S_B     <eps>   3.535568
2122933 67      AA0_B   <eps>   9.55175686
2122933 68      AA1_B   <eps>   5.00592899
2122933 69      AA1_S   <eps>   10.8705435
2122933 70      AA2_B   <eps>   8.441535
2122933 71      #0      <eps>   7.18324804
```

## Making CLG.fst

https://github.com/kaldi-asr/kaldi/blob/0cf2e23a7d8610e18b7f55ce4e09771b92989477/egs/wsj/s5/utils/mkgraph.sh#L113
```
clg=$lang/tmp/CLG_${N}_${P}.fst
clg_tmp=$clg.$$
ilabels=$lang/tmp/ilabels_${N}_${P}
ilabels_tmp=$ilabels.$$
fstcomposecontext $nonterm_opt --context-size=$N --central-position=$P \
    --read-disambig-syms=$lang/phones/disambig.int \
    --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
    $ilabels_tmp $lang/tmp/LG.fst |\
    fstarcsort --sort_type=ilabel > $clg_tmp
mv $clg_tmp $clg
mv $ilabels_tmp $ilabels

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp_test_tgpr/tmp/CLG_1_0.fst | head
0       1       SIL     <eps>   0.698661923
0       2       SIL_B   <UNK>   4.45802927
0       3       SIL_E   <eps>   5.54836178
0       4       SIL_I   <eps>   5.29441261
0       5       SIL_S   <eps>   6.90070009
0       6       SPN     <eps>   11.958147
0       7       SPN_B   <eps>   7.98114967
0       8       SPN_E   <eps>   4.89023113
0       9       SPN_I   AVIATION        13.7248697
0       10      SPN_S   <eps>   7.19705915

fstprint --isymbols=data/lang_nosp/phones.txt --osymbols=data/lang_nosp/words.txt data/lang_nosp_test_tgpr/tmp/CLG_1_0.fst | tail
2122926 2033545 ZH_B    <eps>   -0.104465112
2122927 2122930 ER2_I   <eps>   -0.104464799
2122928 2122846 AW2_E   <eps>   0.588683307
2122928 2122846 K_E     <eps>   0.588683307
2122929 2122879 AH1_B   INCORPORATED    -0.0767181143
2122929 2122880 AH1_E   INCORPORATED'S  3.49405289
2122930 2122931 AH_I    <eps>   -0.104464665
2122931 2122932 IY1_E   <eps>   -0.104464531
2122932 2122933 AW1_I   <eps>   -0.104464404
2122933 2112302 K_E     <eps>   -0.104464293
```

## Making Ha.fst

https://github.com/kaldi-asr/kaldi/blob/0cf2e23a7d8610e18b7f55ce4e09771b92989477/egs/wsj/s5/utils/mkgraph.sh#L126
```
make-h-transducer $nonterm_opt --disambig-syms-out=$dir/disambig_tid.int \
    --transition-scale=$tscale $lang/tmp/ilabels_${N}_${P} $tree $model \
     > $dir/Ha.fst.$$  || exit 1;
mv $dir/Ha.fst.$$ $dir/Ha.fst

tree-info exp/mono0a/tree 
num-pdfs 132
context-width 1
central-position 0

gmm-info exp/mono0a/final.mdl 
number of phones 351
number of pdfs 132
number of transition-ids 2286
number of transition-states 1083
feature dimension 39
number of gaussians 977

fstprint exp/mono0a/graph_nosp_tgpr/Ha.fst | head
0       1       0       1
0       7       0       2
0       13      272     3
0       16      296     4       1.1920929e-07
0       19      320     5
0       22      368     6       -5.96046448e-08
0       25      392     7
0       28      410     8
0       31      416     9       1.1920929e-07
0       34      440     10      5.96046448e-08

fstprint exp/mono0a/graph_nosp_tgpr/Ha.fst | tail
605     0       0       0
606     607     808     0
607     608     810     0       -1.1920929e-07
608     0       0       0
609     610     664     0
610     611     666     0
611     0       0       0
612     613     1384    0
613     614     1386    0
614     0       0       0
```

## Making HCLGa.fst

https://github.com/kaldi-asr/kaldi/blob/0cf2e23a7d8610e18b7f55ce4e09771b92989477/egs/wsj/s5/utils/mkgraph.sh#L140
```
fsttablecompose $dir/Ha.fst "$clg" | fstdeterminizestar --use-log=true | \
    fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
    fstminimizeencoded > $dir/HCLGa.fst.$$ || exit 1;
mv $dir/HCLGa.fst.$$ $dir/HCLGa.fst

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLGa.fst | head                    
0       1       2       <eps>   1.02148438
0       2       3       <eps>   2.67871094
0       3       4       <eps>   2.67871094
0       4       164     <UNK>   7.96972656
0       5       165     <UNK>   7.96972656
0       6       166     <UNK>   4.51953125
0       7       272     <eps>   5.54882812
0       8       296     <eps>   5.29394531
0       9       320     <eps>   6.90039062
0       10      368     <eps>   11.9580078

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLGa.fst | tail
5921160 54      1712    UNO     14.0527344
5921160 57      1856    <eps>   7.62792969
5921160 59      1898    AWE     15.5390625
5921160 60      1904    <eps>   5.79785156
5921160 63      1976    <eps>   8.57324219
5921160 64      2024    <eps>   8.08984375
5921160 66      2072    <eps>   8.24511719
5921160 71      2216    <eps>   9.55078125
5921160 73      2258    <eps>   10.8701172
5921160 74      2264    <eps>   8.44042969

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLGa.fst | grep HELLO
4704402 4704403 824     HELLO   4.4765625
4968514 4704403 824     HELLO   5.02050781
4968519 4704403 824     HELLO   4.38378906
5073244 5073247 1340    HELLO   5.30664062
5073281 5073267 2132    HELLO   1.58984375
5388045 5388046 956     HELLO   4.84082031
5388045 5073247 1340    HELLO   4.84082031
5533949 4704407 1340    HELLO   8.3671875
5534115 4704409 2132    HELLO   4.06738281
5679744 4704407 1340    HELLO   10.6201172
5680734 4704409 2132    HELLO   7.92773438
5714977 5388046 956     HELLO   1.82226562
5714977 5073247 1340    HELLO   1.82226562
5854255 5854277 1340    HELLO   9.37597656
5861463 5857604 2132    HELLO   2.390625
```

## Making HCLG.fst

https://github.com/kaldi-asr/kaldi/blob/0cf2e23a7d8610e18b7f55ce4e09771b92989477/egs/wsj/s5/utils/mkgraph.sh#L149
```
add-self-loops --self-loop-scale=$loopscale --reorder=true $model $dir/HCLGa.fst | \
    $prepare_grammar_command | \
    fstconvert --fst_type=const > $dir/HCLG.fst.$$ || exit 1;
mv $dir/HCLG.fst.$$ $dir/HCLG.fst

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLG.fst | head 
0       5921161 2       <eps>   1.02148438
0       5921162 3       <eps>   2.67871094
0       5921163 4       <eps>   2.67871094
0       5921164 164     <UNK>   7.96972656
0       5921165 165     <UNK>   7.96972656
0       5921166 166     <UNK>   4.51953125
0       7       272     <eps>   5.54882812
0       8       296     <eps>   5.29394531
0       9       320     <eps>   6.90039062
0       10      368     <eps>   11.9580078

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLG.fst | tail 
8792497 5919945 0       <eps>   0.171368375
8792497 8792497 791     <eps>   0.019869579
8792498 5919945 0       <eps>   0.128789589
8792498 8792498 767     <eps>   0.032275755
8792499 5920181 0       <eps>   0.0780357644
8792499 8792499 887     <eps>   0.0612936094
8792500 5920181 0       <eps>   0.128789589
8792500 8792500 767     <eps>   0.032275755
8792501 5920579 0       <eps>   0.0152997198
8792501 8792501 959     <eps>   0.195285901

fstprint --osymbols=data/lang_nosp/words.txt $dir/HCLG.fst | grep HELLO
4704402 4704403 824     HELLO   4.4765625
4968514 4704403 824     HELLO   5.12463903
4968519 4704403 824     HELLO   4.53912354
5073244 5073247 1340    HELLO   5.37283182
5073281 5073267 2132    HELLO   1.60514343
5388045 5388046 956     HELLO   4.90701151
5388045 5073247 1340    HELLO   4.90701151
5533949 4704407 1340    HELLO   8.43337822
5534115 4704409 2132    HELLO   4.08268261
5679744 4704407 1340    HELLO   10.6863079
5680734 4704409 2132    HELLO   7.94303417
5714977 5388046 956     HELLO   1.8884567
5714977 5073247 1340    HELLO   1.8884567
5854255 5854277 1340    HELLO   9.44216728
5861463 5857604 2132    HELLO   2.4059248
```
