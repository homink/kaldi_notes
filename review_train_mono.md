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
