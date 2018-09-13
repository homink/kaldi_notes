# How to prepare HCLG.fst

Kaldi requires HCLG.fst for the WFST decoding. More details are found in [here](http://kaldi-asr.org/doc/graph.html). We review here step by step with the wsj recipe.


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

The prepared above files are used to make L.fst and L_disambig.fst.

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

It is interesting to see that the word's first phone is mapped to that word such as 'EH1_B   "END-QUOTE' in L.fst and L_disambig.fst.

https://github.com/kaldi-asr/kaldi/blob/6c9c00d5bae8cef4fecda99f5f8a3a6d0439e981/egs/wsj/s5/run.sh#L50
```
local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;
```
