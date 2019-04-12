Let us look how steps/train_mono.sh is implemented.

# gmm-init-mono

```
  $cmd JOB=1 $dir/log/init.log \
    gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
```

Now, let us browse exp/mono0a/log/init.log produced by **gmm-init-mono**. Now we know what inputs are loaded in detail.

```
cat exp/mono0a/log/init.log
# gmm-init-mono --shared-phones=data/lang_nosp/phones/sets.int "--train-feats=ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- | subset-feats --n=10 ark:- ark:-|" data/lang_nosp/topo 39 exp/mono0a/0.mdl exp/mono0a/tree 
# Started at Wed Apr  3 15:07:36 PDT 2019
#
gmm-init-mono --shared-phones=data/lang_nosp/phones/sets.int '--train-feats=ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- | subset-feats --n=10 ark:- ark:-|' data/lang_nosp/topo 39 exp/mono0a/0.mdl exp/mono0a/tree 
add-deltas ark:- ark:- 
apply-cmvn --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- 
subset-feats --n=10 ark:- ark:- 
# Accounting: time=0 threads=1
# Ended (code 0) at Wed Apr  3 15:07:36 PDT 2019, elapsed time 0 seconds
```

We input topology, feture data, etc and we obtain model and tree. **topo** is structed as follows.

```
<Topology>
<TopologyEntry>
<ForPhones>
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.75 <Transition> 1 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.75 <Transition> 2 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.75 <Transition> 3 0.25 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones>
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.25 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 2 <PdfClass> 2 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 3 <PdfClass> 3 <Transition> 1 0.25 <Transition> 2 0.25 <Transition> 3 0.25 <Transition> 4 0.25 </State>
<State> 4 <PdfClass> 4 <Transition> 4 0.75 <Transition> 5 0.25 </State>
<State> 5 </State>
</TopologyEntry>
</Topology>
```

Here is summary information about **tree** and **final.mdl**.
```
../../../src/bin/tree-info exp/mono0a/tree 
num-pdfs 132
context-width 1
central-position 0

../../../src/gmmbin/gmm-info exp/mono0a/final.mdl 
number of phones 351
number of pdfs 132
number of transition-ids 2286
number of transition-states 1083
feature dimension 39
number of gaussians 976
```


# compile-train-graphs

```
  echo "$0: Compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
```
```
cat exp/mono0a/log/compile_graphs.1.log 
# compile-train-graphs --read-disambig-syms=data/lang_nosp/phones/disambig.int exp/mono0a/tree exp/mono0a/0.mdl data/lang_nosp/L.fst "ark:sym2int.pl --map-oov 64 -f 2- data/lang_nosp/words.txt < data/train_si84_2kshort/split10/1/text|" "ark:|gzip -c >exp/mono0a/fsts.1.gz" 
# Started at Wed Apr  3 15:07:36 PDT 2019
#
compile-train-graphs --read-disambig-syms=data/lang_nosp/phones/disambig.int exp/mono0a/tree exp/mono0a/0.mdl data/lang_nosp/L.fst 'ark:sym2int.pl --map-oov 64 -f 2- data/lang_nosp/words.txt < data/train_si84_2kshort/split10/1/text|' 'ark:|gzip -c >exp/mono0a/fsts.1.gz' 
LOG (compile-train-graphs[5.5.201~1-afc5e]:main():compile-train-graphs.cc:147) compile-train-graphs: succeeded for 213 graphs, failed for 0
# Accounting: time=1 threads=1
# Ended (code 0) at Wed Apr  3 15:07:37 PDT 2019, elapsed time 1 seconds
```

# align-equal-compiled

```
  echo "$0: Aligning data equally (pass 0)"
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
    align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
    $dir/0.JOB.acc || exit 1;
```
```
# align-equal-compiled "ark:gunzip -c exp/mono0a/fsts.1.gz|" "ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- |" ark,t:- | gmm-acc-stats-ali --binary=true exp/mono0a/0.mdl "ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- |" ark:- exp/mono0a/0.1.acc 
# Started at Thu Apr  4 15:44:40 PDT 2019
#
gmm-acc-stats-ali --binary=true exp/mono0a/0.mdl 'ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- |' ark:- exp/mono0a/0.1.acc 
align-equal-compiled 'ark:gunzip -c exp/mono0a/fsts.1.gz|' 'ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- | add-deltas  ark:- ark:- |' ark,t:- 
add-deltas ark:- ark:- 
apply-cmvn --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- 
add-deltas ark:- ark:- 
apply-cmvn --utt2spk=ark:data/train_si84_2kshort/split10/1/utt2spk scp:data/train_si84_2kshort/split10/1/cmvn.scp scp:data/train_si84_2kshort/split10/1/feats.scp ark:- 
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:105) Processed 50 utterances; for utterance 013o031b avg. like is -106.575 over 507 frames.
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:105) Processed 100 utterances; for utterance 015c0214 avg. like is -106.779 over 328 frames.
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:105) Processed 150 utterances; for utterance 016o030v avg. like is -111.488 over 358 frames.
LOG (apply-cmvn[5.5.201~1-afc5e]:main():apply-cmvn.cc:162) Applied cepstral mean normalization to 213 utterances, errors on 0
LOG (align-equal-compiled[5.5.201~1-afc5e]:main():align-equal-compiled.cc:107) Success: done 213 utterances.
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:105) Processed 200 utterances; for utterance 018c020k avg. like is -107.727 over 485 frames.
LOG (apply-cmvn[5.5.201~1-afc5e]:main():apply-cmvn.cc:162) Applied cepstral mean normalization to 213 utterances, errors on 0
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:112) Done 213 files, 0 with errors.
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:115) Overall avg like per frame (Gaussian only) = -109.132 over 91739 frames.
LOG (gmm-acc-stats-ali[5.5.201~1-afc5e]:main():gmm-acc-stats-ali.cc:123) Written accs.
# Accounting: time=0 threads=1
# Ended (code 0) at Thu Apr  4 15:44:40 PDT 2019, elapsed time 0 seconds
```

# gmm-est

```
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
```
```
gmm-est --min-gaussian-occupancy=3 --mix-up=132 --power=0.25 exp/mono0a/0.mdl 'gmm-sum-accs - exp/mono0a/0.*.acc|' exp/mono0a/1.mdl 
gmm-sum-accs - exp/mono0a/0.1.acc exp/mono0a/0.10.acc exp/mono0a/0.2.acc exp/mono0a/0.3.acc exp/mono0a/0.4.acc exp/mono0a/0.5.acc exp/mono0a/0.6.acc exp/mono0a/0.7.acc exp/mono0a/0.8.acc exp/mono0a/0.9.acc 
LOG (gmm-sum-accs[5.5.201~1-afc5e]:main():gmm-sum-accs.cc:63) Summed 10 stats, total count 849568, avg like/frame -108.537
LOG (gmm-sum-accs[5.5.201~1-afc5e]:main():gmm-sum-accs.cc:66) Total count of stats is 849568
LOG (gmm-sum-accs[5.5.201~1-afc5e]:main():gmm-sum-accs.cc:67) Written stats to -
LOG (gmm-est[5.5.201~1-afc5e]:MleUpdate():transition-model.cc:528) TransitionModel::Update, objf change is 0.0979067 per frame over 849568 frames. 
LOG (gmm-est[5.5.201~1-afc5e]:MleUpdate():transition-model.cc:531) 0 probabilities floored, 549 out of 1083 transition-states skipped due to insuffient data (it is normal to have some skipped.)
LOG (gmm-est[5.5.201~1-afc5e]:main():gmm-est.cc:102) Transition model update: Overall 0.0979067 log-like improvement per frame over 849568 frames.
LOG (gmm-est[5.5.201~1-afc5e]:MleAmDiagGmmUpdate():mle-am-diag-gmm.cc:225) 0 variance elements floored in 0 Gaussians, out of 132
LOG (gmm-est[5.5.201~1-afc5e]:MleAmDiagGmmUpdate():mle-am-diag-gmm.cc:229) Removed 0 Gaussians due to counts < --min-gaussian-occupancy=3 and --remove-low-count-gaussians=true
LOG (gmm-est[5.5.201~1-afc5e]:main():gmm-est.cc:113) GMM update: Overall 0.427029 objective function improvement per frame over 849568 frames
LOG (gmm-est[5.5.201~1-afc5e]:main():gmm-est.cc:116) GMM update: Overall avg like per frame = -108.537 over 849568 frames.
LOG (gmm-est[5.5.201~1-afc5e]:SplitByCount():am-diag-gmm.cc:116) Split 132 states with target = 132, power = 0.25, perturb_factor = 0.01 and min_count = 20, split #Gauss from 132 to 132
LOG (gmm-est[5.5.201~1-afc5e]:main():gmm-est.cc:146) Written model to exp/mono0a/1.mdl
```

# EM iteration

```
beam=6 # will change to 10 below after 1st pass
# note: using slightly wider beams for WSJ vs. RM.
x=1
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] --careful=$careful "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;

    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  fi
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=10
  x=$[$x+1]
done
```
