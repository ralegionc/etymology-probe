## Data
- English entries walked : 68303
- labelled entries : 36329
- entries with a donor class : 33982
- spellings with conflicting labels (dropped) : 0
- common enough (zipf >= 2.5) : 11540
- dropped classes : Celtic (51), Portuguese (36), Russian (26), Chinese (22), Hebrew (17), Turkish (16), Malay (14), Yiddish (12), Persian (11), Dravidian (9)
- rare words (spelling training only) : 20767
- words : 11326
- classes : {'French': 5030, 'native': 2530, 'Latin': 2264, 'Norse': 296, 'Greek': 237, 'Dutch': 213, 'Spanish': 191, 'Italian': 173, 'German': 139, 'Indic': 107, 'Japanese': 81, 'Arabic': 65}
- kind : {'inferred': 5667, 'explicit': 3129, 'inherited': 2530}

## Baselines (balanced accuracy, balanced log loss)
- prior  8.3%  3.569
- spell  42.9%  1.775
- cnn    42.1%  1.707
- freq   19.5%  2.204
- sem    34.7%  1.948
- N      43.9%  1.694
- S      52.2%  1.446
- SF     53.5%  1.406
- SS     55.5%  1.303
- B      56.6%  1.285

## Headline, layer chosen inside each training fold

### OLMo-1B-0724-hf@main
- H    vs N: 56.3% (+12.4 points [+9.9, +14.7]; +0.357 nats [+0.303, +0.410]), layers [13, 13, 13, 13, 13]
- N+H  vs N: 59.4% (+15.4 points [+12.9, +17.7]; +0.436 nats [+0.391, +0.482]), layers [13, 11, 15, 13, 13]
- S+H  vs S: 61.6% (+9.5 points [+7.5, +11.5]; +0.279 nats [+0.242, +0.313]), layers [13, 13, 13, 13, 13]
- B+H  vs B: 61.8% (+5.2 points [+3.5, +6.7]; +0.148 nats [+0.121, +0.175]), layers [13, 11, 13, 13, 13]

### OLMo-2-0425-1B@main
- H    vs N: 58.9% (+14.9 points [+12.3, +17.3]; +0.400 nats [+0.341, +0.457]), layers [12, 12, 12, 12, 12]
- N+H  vs N: 60.7% (+16.8 points [+14.5, +18.8]; +0.479 nats [+0.428, +0.526]), layers [16, 11, 16, 16, 6]
- S+H  vs S: 63.0% (+10.8 points [+8.9, +12.7]; +0.302 nats [+0.261, +0.337]), layers [16, 16, 6, 16, 6]
- B+H  vs B: 62.5% (+5.9 points [+4.3, +7.7]; +0.174 nats [+0.143, +0.206]), layers [16, 12, 12, 12, 6]

### pythia-1.4b@main
- H    vs N: 57.9% (+13.9 points [+11.3, +16.5]; +0.350 nats [+0.293, +0.408]), layers [16, 19, 16, 18, 18]
- N+H  vs N: 60.8% (+16.8 points [+14.5, +18.9]; +0.456 nats [+0.405, +0.507]), layers [22, 19, 16, 22, 22]
- S+H  vs S: 61.9% (+9.7 points [+7.7, +11.7]; +0.287 nats [+0.247, +0.325]), layers [22, 22, 16, 22, 22]
- B+H  vs B: 62.5% (+5.9 points [+4.2, +7.5]; +0.158 nats [+0.130, +0.189]), layers [22, 22, 16, 22, 22]

### pythia-160m@main
- H    vs N: 47.2% (+3.3 points [+0.5, +5.9]; +0.051 nats [-0.003, +0.106]), layers [9, 8, 7, 6, 9]
- N+H  vs N: 54.3% (+10.4 points [+8.3, +12.4]; +0.258 nats [+0.220, +0.293]), layers [11, 11, 11, 11, 11]
- S+H  vs S: 57.9% (+5.7 points [+4.0, +7.5]; +0.124 nats [+0.097, +0.149]), layers [7, 11, 7, 11, 11]
- B+H  vs B: 58.0% (+1.4 points [-0.1, +2.8]; +0.045 nats [+0.029, +0.061]), layers [7, 7, 7, 6, 11]

### pythia-1b@main
- H    vs N: 55.7% (+11.7 points [+9.0, +14.2]; +0.306 nats [+0.251, +0.360]), layers [12, 12, 12, 10, 10]
- N+H  vs N: 58.0% (+14.0 points [+11.6, +16.2]; +0.420 nats [+0.375, +0.466]), layers [14, 14, 14, 13, 13]
- S+H  vs S: 60.4% (+8.2 points [+6.1, +10.2]; +0.251 nats [+0.218, +0.283]), layers [14, 14, 14, 14, 13]
- B+H  vs B: 61.4% (+4.8 points [+3.1, +6.5]; +0.132 nats [+0.109, +0.156]), layers [14, 14, 14, 14, 13]

### pythia-2.8b@main
- H    vs N: 57.4% (+13.4 points [+10.8, +16.0]; +0.381 nats [+0.321, +0.440]), layers [21, 21, 21, 21, 21]
- N+H  vs N: 60.8% (+16.9 points [+14.5, +19.0]; +0.466 nats [+0.416, +0.517]), layers [21, 21, 21, 29, 28]
- S+H  vs S: 62.1% (+9.9 points [+7.9, +11.8]; +0.282 nats [+0.242, +0.319]), layers [23, 21, 21, 29, 28]
- B+H  vs B: 61.9% (+5.3 points [+3.6, +7.0]; +0.158 nats [+0.127, +0.189]), layers [23, 21, 21, 21, 25]

### pythia-410m@main
- H    vs N: 53.9% (+9.9 points [+7.3, +12.5]; +0.251 nats [+0.191, +0.311]), layers [21, 21, 21, 20, 21]
- N+H  vs N: 58.6% (+14.7 points [+12.3, +16.8]; +0.390 nats [+0.345, +0.435]), layers [21, 21, 21, 20, 21]
- S+H  vs S: 61.4% (+9.2 points [+7.4, +11.1]; +0.235 nats [+0.200, +0.268]), layers [21, 21, 21, 21, 21]
- B+H  vs B: 61.0% (+4.4 points [+2.8, +5.9]; +0.121 nats [+0.097, +0.144]), layers [21, 21, 21, 21, 21]

### pythia-410m@step0
- H    vs N: 15.3% (-28.6 points [-31.1, -26.1]; -0.755 nats [-0.811, -0.701]), layers [5, 7, 7, 7, 6]
- N+H  vs N: 44.1% (+0.1 points [-0.5, +0.7]; +0.001 nats [-0.004, +0.005]), layers [7, 7, 6, 7, 6]
- S+H  vs S: 51.9% (-0.3 points [-1.0, +0.3]; -0.000 nats [-0.003, +0.003]), layers [6, 5, 6, 7, 6]
- B+H  vs B: 56.1% (-0.5 points [-1.1, +0.0]; -0.000 nats [-0.003, +0.001]), layers [4, 7, 7, 7, 6]

### pythia-410m@step2000
- H    vs N: 38.6% (-5.3 points [-8.0, -3.1]; -0.138 nats [-0.191, -0.087]), layers [23, 23, 23, 22, 23]
- N+H  vs N: 46.8% (+2.8 points [+1.4, +4.3]; +0.134 nats [+0.109, +0.157]), layers [23, 19, 23, 22, 23]
- S+H  vs S: 53.5% (+1.4 points [+0.1, +2.8]; +0.062 nats [+0.046, +0.080]), layers [22, 19, 23, 22, 19]
- B+H  vs B: 57.1% (+0.5 points [-0.4, +1.4]; +0.022 nats [+0.010, +0.033]), layers [23, 17, 19, 19, 19]

### pythia-410m@step32000
- H    vs N: 52.5% (+8.6 points [+5.8, +11.2]; +0.211 nats [+0.157, +0.269]), layers [17, 19, 19, 16, 19]
- N+H  vs N: 57.2% (+13.3 points [+11.1, +15.3]; +0.356 nats [+0.313, +0.398]), layers [20, 19, 19, 20, 19]
- S+H  vs S: 59.8% (+7.6 points [+5.8, +9.5]; +0.204 nats [+0.171, +0.232]), layers [20, 19, 20, 20, 19]
- B+H  vs B: 60.2% (+3.6 points [+2.1, +5.0]; +0.096 nats [+0.073, +0.118]), layers [17, 17, 20, 17, 19]

### pythia-410m@step512
- H    vs N: 25.4% (-18.5 points [-20.9, -16.1]; -0.482 nats [-0.532, -0.430]), layers [20, 21, 20, 20, 15]
- N+H  vs N: 44.6% (+0.7 points [-0.6, +1.8]; +0.039 nats [+0.025, +0.054]), layers [20, 21, 20, 20, 8]
- S+H  vs S: 52.2% (+0.0 points [-1.0, +1.0]; +0.015 nats [+0.006, +0.024]), layers [20, 20, 20, 20, 15]
- B+H  vs B: 56.6% (-0.0 points [-0.6, +0.6]; +0.001 nats [-0.003, +0.005]), layers [17, 12, 8, 17, 8]

### pythia-410m@step8000
- H    vs N: 46.8% (+2.9 points [+0.3, +5.3]; +0.114 nats [+0.062, +0.168]), layers [22, 22, 22, 22, 22]
- N+H  vs N: 52.9% (+8.9 points [+7.1, +10.7]; +0.281 nats [+0.246, +0.317]), layers [22, 22, 22, 22, 22]
- S+H  vs S: 57.4% (+5.2 points [+3.8, +6.7]; +0.152 nats [+0.128, +0.176]), layers [22, 22, 22, 22, 22]
- B+H  vs B: 58.1% (+1.5 points [+0.4, +2.8]; +0.066 nats [+0.050, +0.082]), layers [22, 22, 22, 22, 22]

### pythia-70m@main
- H    vs N: 40.9% (-3.0 points [-5.7, -0.6]; -0.038 nats [-0.093, +0.017]), layers [4, 4, 4, 4, 4]
- N+H  vs N: 49.2% (+5.3 points [+3.6, +7.0]; +0.182 nats [+0.155, +0.211]), layers [4, 4, 4, 4, 4]
- S+H  vs S: 53.7% (+1.5 points [+0.0, +2.9]; +0.079 nats [+0.059, +0.096]), layers [3, 4, 4, 4, 4]
- B+H  vs B: 56.8% (+0.2 points [-0.8, +1.3]; +0.028 nats [+0.018, +0.039]), layers [3, 4, 4, 4, 4]

## Mean pooling instead of last token
- pythia-1.4b@main H: 53.7% (+9.8 points [+7.3, +12.1])
- pythia-1.4b@main B+H: 60.7% (+4.1 points [+2.7, +5.5])
- pythia-410m@main H: 49.7% (+5.7 points [+3.1, +8.2])
- pythia-410m@main B+H: 60.2% (+3.6 points [+2.2, +5.0])

## Per class, pythia-2.8b@main
- Spanish   n=  191 recall 63.9% -> 77.0%  gain +0.297 nats
- Indic     n=  107 recall 56.1% -> 65.4%  gain +0.296 nats
- German    n=  139 recall 53.2% -> 55.4%  gain +0.215 nats
- Italian   n=  173 recall 61.8% -> 71.7%  gain +0.203 nats
- Japanese  n=   81 recall 66.7% -> 75.3%  gain +0.178 nats
- Greek     n=  237 recall 69.2% -> 72.6%  gain +0.130 nats
- Latin     n= 2264 recall 51.5% -> 55.3%  gain +0.118 nats
- native    n= 2530 recall 57.8% -> 62.3%  gain +0.106 nats
- French    n= 5030 recall 45.9% -> 50.2%  gain +0.099 nats
- Arabic    n=   65 recall 61.5% -> 64.6%  gain +0.097 nats
- Dutch     n=  213 recall 44.1% -> 46.9%  gain +0.096 nats
- Norse     n=  296 recall 47.3% -> 46.3%  gain +0.065 nats

## Pairs (AUC [95% CI]), pythia-2.8b@main
- Dutch vs native: N 0.815 [0.786, 0.840], S 0.838 [0.808, 0.864], B 0.866 [0.841, 0.890], H 0.853 [0.824, 0.875], B+H 0.887 [0.864, 0.909]
- Norse vs native: N 0.755 [0.726, 0.784], S 0.777 [0.752, 0.803], B 0.789 [0.764, 0.815], H 0.706 [0.675, 0.734], B+H 0.785 [0.762, 0.807]
  Dutch vs native by model:
model                      B    B+H      H      N      S
tag                                                     
OLMo-1B-0724-hf@main   0.866  0.876  0.832  0.815  0.838
OLMo-2-0425-1B@main    0.866  0.888  0.858  0.815  0.838
pythia-1.4b@main       0.866  0.886  0.845  0.815  0.838
pythia-160m@main       0.866  0.868  0.789  0.815  0.838
pythia-1b@main         0.866  0.884  0.836  0.815  0.838
pythia-2.8b@main       0.866  0.887  0.853  0.815  0.838
pythia-410m@main       0.866  0.879  0.826  0.815  0.838
pythia-410m@step0      0.866  0.866  0.540  0.815  0.838
pythia-410m@step2000   0.866  0.870  0.759  0.815  0.838
pythia-410m@step32000  0.866  0.875  0.814  0.815  0.838
pythia-410m@step512    0.866  0.868  0.657  0.815  0.838
pythia-410m@step8000   0.866  0.872  0.788  0.815  0.838
pythia-70m@main        0.866  0.871  0.783  0.815  0.838
  Norse vs native by model:
model                      B    B+H      H      N      S
tag                                                     
OLMo-1B-0724-hf@main   0.789  0.797  0.735  0.755  0.777
OLMo-2-0425-1B@main    0.789  0.795  0.746  0.755  0.777
pythia-1.4b@main       0.789  0.791  0.727  0.755  0.777
pythia-160m@main       0.789  0.794  0.700  0.755  0.777
pythia-1b@main         0.789  0.788  0.708  0.755  0.777
pythia-2.8b@main       0.789  0.785  0.706  0.755  0.777
pythia-410m@main       0.789  0.794  0.703  0.755  0.777
pythia-410m@step0      0.789  0.788  0.491  0.755  0.777
pythia-410m@step2000   0.789  0.788  0.650  0.755  0.777
pythia-410m@step32000  0.789  0.796  0.719  0.755  0.777
pythia-410m@step512    0.789  0.788  0.579  0.755  0.777
pythia-410m@step8000   0.789  0.786  0.670  0.755  0.777
pythia-70m@main        0.789  0.790  0.673  0.755  0.777

## Disguised loans (spelling says native), pythia-2.8b@main
- n=365 by class {'French': 231, 'Norse': 51, 'Latin': 41, 'Dutch': 18, 'German': 8, 'Indic': 6, 'Spanish': 4, 'Japanese': 3, 'Greek': 2, 'Italian': 1}
- right donor: B 11.2%, B+H 22.2%, mean gain +0.230 nats
- recovered by the hidden states (43): low (Norse), view (French), style (French), simple (French), text (French), sort (French), award (French), benefit (French), apply (French), treat (French), attend (French), theatre (French), reply (French), reward (French), acknowledge (French), drill (Dutch), blend (Norse), exceed (French), tomb (French), wardrobe (French), render (French), yen (Japanese), forestry (French), refresh (French), amend (French), redeem (French), sly (Norse), prowess (French), medley (French), horde (French), melee (French), ethereal (Latin), lien (French), tout (French), sever (French), lard (French), belvedere (Italian), chutney (Indic), hombre (Spanish), boreal (Latin)
- lost (3): fame, foolhardy, sacred
  they     Norse   B->native   B+H->native   logp -1.66 -> -1.43
  them     Norse   B->native   B+H->native   logp -1.29 -> -1.37
  sky      Norse   B->Norse    B+H->Norse    logp -0.34 -> -0.48
  egg      Norse   B->Norse    B+H->Norse    logp -0.62 -> -0.64
  knife    Norse   B->native   B+H->native   logp -2.54 -> -2.36
  take     Norse   B->Norse    B+H->Norse    logp -0.10 -> -0.23
  get      Norse   B->Norse    B+H->Norse    logp -0.78 -> -1.00
  give     Norse   B->Norse    B+H->Norse    logp -0.86 -> -0.66
  window   Norse   B->native   B+H->native   logp -2.82 -> -2.03
  husband  Norse   B->German   B+H->native   logp -2.74 -> -2.86
  law      Norse   B->Norse    B+H->Norse    logp -0.19 -> -0.43
  wrong    Norse   B->native   B+H->native   logp -1.82 -> -1.30
  ugly     Norse   B->native   B+H->native   logp -1.70 -> -1.63
  boss     Dutch   B->Dutch    B+H->Dutch    logp -0.87 -> -0.58
  cookie   Dutch   B->Dutch    B+H->Dutch    logp -0.14 -> -0.17
  yacht    Dutch   B->Dutch    B+H->Dutch    logp -0.85 -> -0.87
  skipper  Dutch   B->Dutch    B+H->Dutch    logp -0.18 -> -0.18

## Explicit labels only
- subset: {'n': 5659, 'classes': {'native': 2530, 'French': 1314, 'Latin': 1060, 'Spanish': 170, 'Italian': 135, 'German': 104, 'Indic': 80, 'Dutch': 72, 'Japanese': 68, 'Greek': 66, 'Arabic': 42, 'Norse': 18}}
- OLMo-1B-0724-hf@main B+H vs B: +4.4 points [+1.8, +7.1]
- OLMo-1B-0724-hf@main N+H vs N: +16.9 points [+13.3, +20.3]
- OLMo-2-0425-1B@main B+H vs B: +5.0 points [+2.1, +7.9]
- OLMo-2-0425-1B@main N+H vs N: +17.7 points [+14.0, +21.3]
- pythia-1.4b@main B+H vs B: +6.2 points [+3.4, +9.0]
- pythia-1.4b@main N+H vs N: +18.0 points [+14.8, +20.9]
- pythia-160m@main B+H vs B: +1.3 points [-0.6, +3.4]
- pythia-160m@main N+H vs N: +10.1 points [+7.1, +12.9]
- pythia-1b@main B+H vs B: +4.3 points [+1.4, +7.0]
- pythia-1b@main N+H vs N: +14.6 points [+10.9, +18.3]
- pythia-2.8b@main B+H vs B: +4.1 points [+1.3, +6.8]
- pythia-2.8b@main N+H vs N: +17.1 points [+13.5, +20.6]
- pythia-410m@main B+H vs B: +4.9 points [+2.5, +7.7]
- pythia-410m@main N+H vs N: +14.9 points [+11.4, +18.3]
- pythia-410m@step0 B+H vs B: +0.0 points [-0.7, +0.7]
- pythia-410m@step0 N+H vs N: +0.3 points [-0.4, +0.9]
- pythia-410m@step2000 B+H vs B: +0.7 points [-0.8, +2.6]
- pythia-410m@step2000 N+H vs N: +2.6 points [+0.0, +4.9]
- pythia-410m@step32000 B+H vs B: +2.7 points [+0.2, +5.3]
- pythia-410m@step32000 N+H vs N: +11.2 points [+7.7, +14.6]
- pythia-410m@step512 B+H vs B: +0.5 points [-0.4, +1.6]
- pythia-410m@step512 N+H vs N: +1.3 points [-0.6, +3.1]
- pythia-410m@step8000 B+H vs B: +1.0 points [-1.3, +3.2]
- pythia-410m@step8000 N+H vs N: +7.5 points [+4.3, +10.4]
- pythia-70m@main B+H vs B: +0.3 points [-1.4, +1.7]
- pythia-70m@main N+H vs N: +4.6 points [+1.8, +7.4]

## Mentions
- pythia-1.4b@main B+H vs B (n=3302): Intercept +0.3414 (se 0.0947, p 0.000309), log_true +0.0723 (se 0.0159, p 5.64e-06), log_placebo -0.0117 (se 0.0139, p 0.4), log_count -0.0607 (se 0.0142, p 1.87e-05)
- pythia-1.4b@main B+H vs B, loans only (n=2702): Intercept +0.4044 (se 0.1023, p 7.7e-05), log_true +0.0846 (se 0.0202, p 2.84e-05), log_placebo -0.0257 (se 0.0170, p 0.13), log_count -0.0755 (se 0.0175, p 1.61e-05)
- pythia-1.4b@main B+H vs B, exact counts (n=1744): Intercept +0.5870 (se 0.1492, p 8.3e-05), log_true +0.1582 (se 0.0353, p 7.24e-06), log_placebo -0.0017 (se 0.0330, p 0.96), log_count -0.1322 (se 0.0322, p 4.15e-05)
- pythia-1.4b@main S+H vs S (n=3302): Intercept +0.1753 (se 0.1294, p 0.175), log_true +0.0984 (se 0.0208, p 2.35e-06), log_placebo -0.0050 (se 0.0184, p 0.786), log_count +0.0024 (se 0.0178, p 0.893)
- pythia-1.4b@main S+H vs S, loans only (n=2702): Intercept +0.3463 (se 0.1357, p 0.0107), log_true +0.1213 (se 0.0252, p 1.53e-06), log_placebo -0.0257 (se 0.0216, p 0.234), log_count -0.0373 (se 0.0213, p 0.0795)
- pythia-1.4b@main S+H vs S, exact counts (n=1744): Intercept +0.5693 (se 0.1887, p 0.00255), log_true +0.2224 (se 0.0450, p 7.61e-07), log_placebo +0.0053 (se 0.0413, p 0.898), log_count -0.0976 (se 0.0396, p 0.0137)
- pythia-160m@main B+H vs B (n=3302): Intercept -0.1184 (se 0.0690, p 0.0862), log_true +0.0326 (se 0.0094, p 0.000521), log_placebo -0.0174 (se 0.0079, p 0.0275), log_count +0.0018 (se 0.0084, p 0.835)
- pythia-160m@main B+H vs B, loans only (n=2702): Intercept -0.0868 (se 0.0726, p 0.232), log_true +0.0399 (se 0.0120, p 0.000849), log_placebo -0.0229 (se 0.0094, p 0.0149), log_count -0.0059 (se 0.0105, p 0.575)
- pythia-160m@main B+H vs B, exact counts (n=1744): Intercept -0.1283 (se 0.0954, p 0.179), log_true +0.0475 (se 0.0189, p 0.0121), log_placebo -0.0224 (se 0.0180, p 0.212), log_count -0.0037 (se 0.0186, p 0.84)
- pythia-160m@main S+H vs S (n=3302): Intercept -0.3443 (se 0.1009, p 0.000642), log_true +0.0501 (se 0.0151, p 0.000923), log_placebo -0.0110 (se 0.0130, p 0.399), log_count +0.0504 (se 0.0136, p 0.000204)
- pythia-160m@main S+H vs S, loans only (n=2702): Intercept -0.2124 (se 0.1053, p 0.0437), log_true +0.0662 (se 0.0186, p 0.000376), log_placebo -0.0237 (se 0.0149, p 0.111), log_count +0.0199 (se 0.0164, p 0.225)
- pythia-160m@main S+H vs S, exact counts (n=1744): Intercept -0.2734 (se 0.1383, p 0.048), log_true +0.0762 (se 0.0303, p 0.0118), log_placebo -0.0249 (se 0.0277, p 0.37), log_count +0.0320 (se 0.0289, p 0.269)
- pythia-1b@main B+H vs B (n=3302): Intercept +0.2295 (se 0.0793, p 0.00382), log_true +0.0522 (se 0.0142, p 0.000227), log_placebo -0.0253 (se 0.0127, p 0.0468), log_count -0.0253 (se 0.0124, p 0.0418)
- pythia-1b@main B+H vs B, loans only (n=2702): Intercept +0.2821 (se 0.0873, p 0.00123), log_true +0.0668 (se 0.0175, p 0.000129), log_placebo -0.0393 (se 0.0151, p 0.00922), log_count -0.0381 (se 0.0153, p 0.0128)
- pythia-1b@main B+H vs B, exact counts (n=1744): Intercept +0.3138 (se 0.1280, p 0.0142), log_true +0.0924 (se 0.0286, p 0.00124), log_placebo -0.0440 (se 0.0284, p 0.121), log_count -0.0467 (se 0.0271, p 0.0845)
- pythia-1b@main S+H vs S (n=3302): Intercept +0.0738 (se 0.1115, p 0.508), log_true +0.0758 (se 0.0192, p 8.15e-05), log_placebo -0.0188 (se 0.0177, p 0.289), log_count +0.0344 (se 0.0167, p 0.0389)
- pythia-1b@main S+H vs S, loans only (n=2702): Intercept +0.2339 (se 0.1191, p 0.0494), log_true +0.1018 (se 0.0225, p 5.88e-06), log_placebo -0.0404 (se 0.0204, p 0.0476), log_count -0.0032 (se 0.0199, p 0.874)
- pythia-1b@main S+H vs S, exact counts (n=1744): Intercept +0.2752 (se 0.1676, p 0.101), log_true +0.1430 (se 0.0369, p 0.000104), log_placebo -0.0481 (se 0.0363, p 0.185), log_count -0.0060 (se 0.0346, p 0.862)
- pythia-2.8b@main B+H vs B (n=3302): Intercept +0.3383 (se 0.1014, p 0.000853), log_true +0.0739 (se 0.0160, p 3.7e-06), log_placebo -0.0121 (se 0.0143, p 0.396), log_count -0.0618 (se 0.0146, p 2.36e-05)
- pythia-2.8b@main B+H vs B, loans only (n=2702): Intercept +0.4147 (se 0.1084, p 0.00013), log_true +0.0954 (se 0.0196, p 1.19e-06), log_placebo -0.0291 (se 0.0169, p 0.0842), log_count -0.0806 (se 0.0178, p 5.92e-06)
- pythia-2.8b@main B+H vs B, exact counts (n=1744): Intercept +0.6165 (se 0.1501, p 3.99e-05), log_true +0.1545 (se 0.0324, p 1.83e-06), log_placebo -0.0295 (se 0.0321, p 0.359), log_count -0.1316 (se 0.0320, p 3.98e-05)
- pythia-2.8b@main S+H vs S (n=3302): Intercept +0.1566 (se 0.1374, p 0.254), log_true +0.1059 (se 0.0208, p 3.65e-07), log_placebo -0.0020 (se 0.0192, p 0.916), log_count -0.0022 (se 0.0188, p 0.909)
- pythia-2.8b@main S+H vs S, loans only (n=2702): Intercept +0.3386 (se 0.1431, p 0.018), log_true +0.1398 (se 0.0245, p 1.21e-08), log_placebo -0.0250 (se 0.0219, p 0.255), log_count -0.0455 (se 0.0221, p 0.04)
- pythia-2.8b@main S+H vs S, exact counts (n=1744): Intercept +0.6254 (se 0.1908, p 0.00104), log_true +0.2385 (se 0.0410, p 6.18e-09), log_placebo -0.0084 (se 0.0406, p 0.836), log_count -0.1095 (se 0.0393, p 0.00532)
- pythia-410m@main B+H vs B (n=3302): Intercept +0.0995 (se 0.0836, p 0.234), log_true +0.0450 (se 0.0133, p 0.000716), log_placebo -0.0114 (se 0.0115, p 0.324), log_count -0.0191 (se 0.0124, p 0.123)
- pythia-410m@main B+H vs B, loans only (n=2702): Intercept +0.1701 (se 0.0903, p 0.0597), log_true +0.0603 (se 0.0166, p 0.000278), log_placebo -0.0204 (se 0.0136, p 0.134), log_count -0.0361 (se 0.0152, p 0.0176)
- pythia-410m@main B+H vs B, exact counts (n=1744): Intercept +0.1955 (se 0.1272, p 0.124), log_true +0.0815 (se 0.0271, p 0.00262), log_placebo -0.0324 (se 0.0260, p 0.213), log_count -0.0428 (se 0.0268, p 0.111)
- pythia-410m@main S+H vs S (n=3302): Intercept -0.0904 (se 0.1189, p 0.447), log_true +0.0689 (se 0.0184, p 0.000182), log_placebo -0.0044 (se 0.0166, p 0.792), log_count +0.0405 (se 0.0165, p 0.0141)
- pythia-410m@main S+H vs S, loans only (n=2702): Intercept +0.0877 (se 0.1239, p 0.479), log_true +0.0960 (se 0.0217, p 9.29e-06), log_placebo -0.0200 (se 0.0187, p 0.285), log_count -0.0015 (se 0.0195, p 0.939)
- pythia-410m@main S+H vs S, exact counts (n=1744): Intercept +0.1094 (se 0.1677, p 0.514), log_true +0.1268 (se 0.0357, p 0.000385), log_placebo -0.0367 (se 0.0336, p 0.275), log_count -0.0004 (se 0.0348, p 0.991)
- pythia-410m@step0 B+H vs B (n=3302): Intercept -0.0223 (se 0.0076, p 0.00352), log_true +0.0002 (se 0.0011, p 0.838), log_placebo -0.0017 (se 0.0010, p 0.0788), log_count +0.0040 (se 0.0010, p 0.000135)
- pythia-410m@step0 B+H vs B, loans only (n=2702): Intercept -0.0216 (se 0.0083, p 0.00929), log_true +0.0000 (se 0.0014, p 0.996), log_placebo -0.0021 (se 0.0012, p 0.0892), log_count +0.0039 (se 0.0013, p 0.00357)
- pythia-410m@step0 B+H vs B, exact counts (n=1744): Intercept -0.0346 (se 0.0120, p 0.00406), log_true -0.0005 (se 0.0025, p 0.827), log_placebo +0.0004 (se 0.0026, p 0.891), log_count +0.0067 (se 0.0025, p 0.00749)
- pythia-410m@step0 S+H vs S (n=3302): Intercept -0.0351 (se 0.0107, p 0.000998), log_true -0.0002 (se 0.0014, p 0.907), log_placebo -0.0014 (se 0.0013, p 0.269), log_count +0.0062 (se 0.0013, p 6.63e-07)
- pythia-410m@step0 S+H vs S, loans only (n=2702): Intercept -0.0285 (se 0.0113, p 0.0117), log_true +0.0003 (se 0.0018, p 0.881), log_placebo -0.0017 (se 0.0015, p 0.27), log_count +0.0047 (se 0.0016, p 0.00261)
- pythia-410m@step0 S+H vs S, exact counts (n=1744): Intercept -0.0305 (se 0.0153, p 0.0456), log_true -0.0008 (se 0.0031, p 0.793), log_placebo -0.0004 (se 0.0034, p 0.902), log_count +0.0052 (se 0.0029, p 0.0748)
- pythia-410m@step2000 B+H vs B (n=3302): Intercept -0.1085 (se 0.0438, p 0.0134), log_true +0.0094 (se 0.0067, p 0.157), log_placebo -0.0122 (se 0.0060, p 0.0432), log_count +0.0194 (se 0.0061, p 0.00147)
- pythia-410m@step2000 B+H vs B, loans only (n=2702): Intercept -0.1133 (se 0.0470, p 0.0158), log_true +0.0045 (se 0.0084, p 0.592), log_placebo -0.0138 (se 0.0072, p 0.0543), log_count +0.0211 (se 0.0075, p 0.00489)
- pythia-410m@step2000 B+H vs B, exact counts (n=1744): Intercept -0.1000 (se 0.0613, p 0.103), log_true +0.0047 (se 0.0132, p 0.724), log_placebo -0.0123 (se 0.0135, p 0.363), log_count +0.0178 (se 0.0126, p 0.159)
- pythia-410m@step2000 S+H vs S (n=3302): Intercept -0.3069 (se 0.0626, p 9.43e-07), log_true +0.0109 (se 0.0103, p 0.29), log_placebo -0.0078 (se 0.0096, p 0.421), log_count +0.0697 (se 0.0090, p 8.48e-15)
- pythia-410m@step2000 S+H vs S, loans only (n=2702): Intercept -0.2661 (se 0.0661, p 5.68e-05), log_true +0.0029 (se 0.0119, p 0.804), log_placebo -0.0124 (se 0.0111, p 0.263), log_count +0.0619 (se 0.0105, p 4.48e-09)
- pythia-410m@step2000 S+H vs S, exact counts (n=1744): Intercept -0.1847 (se 0.0882, p 0.0363), log_true +0.0005 (se 0.0193, p 0.98), log_placebo -0.0204 (se 0.0199, p 0.304), log_count +0.0481 (se 0.0177, p 0.0066)
- pythia-410m@step32000 B+H vs B (n=3302): Intercept +0.0439 (se 0.0771, p 0.569), log_true +0.0430 (se 0.0127, p 0.000711), log_placebo -0.0166 (se 0.0114, p 0.144), log_count -0.0030 (se 0.0117, p 0.801)
- pythia-410m@step32000 B+H vs B, loans only (n=2702): Intercept +0.0936 (se 0.0827, p 0.258), log_true +0.0543 (se 0.0158, p 0.000599), log_placebo -0.0217 (se 0.0136, p 0.111), log_count -0.0151 (se 0.0144, p 0.296)
- pythia-410m@step32000 B+H vs B, exact counts (n=1744): Intercept +0.1301 (se 0.1167, p 0.265), log_true +0.0722 (se 0.0264, p 0.00623), log_placebo -0.0308 (se 0.0252, p 0.221), log_count -0.0320 (se 0.0263, p 0.223)
- pythia-410m@step32000 S+H vs S (n=3302): Intercept -0.1777 (se 0.1087, p 0.102), log_true +0.0584 (se 0.0178, p 0.00104), log_placebo -0.0109 (se 0.0164, p 0.505), log_count +0.0600 (se 0.0158, p 0.000155)
- pythia-410m@step32000 S+H vs S, loans only (n=2702): Intercept -0.0389 (se 0.1134, p 0.731), log_true +0.0763 (se 0.0207, p 0.000237), log_placebo -0.0212 (se 0.0185, p 0.253), log_count +0.0276 (se 0.0187, p 0.141)
- pythia-410m@step32000 S+H vs S, exact counts (n=1744): Intercept +0.0425 (se 0.1562, p 0.786), log_true +0.1121 (se 0.0347, p 0.00123), log_placebo -0.0311 (se 0.0328, p 0.343), log_count +0.0057 (se 0.0350, p 0.87)
- pythia-410m@step512 B+H vs B (n=3302): Intercept -0.0468 (se 0.0165, p 0.00457), log_true +0.0024 (se 0.0022, p 0.262), log_placebo -0.0064 (se 0.0019, p 0.000852), log_count +0.0067 (se 0.0021, p 0.00119)
- pythia-410m@step512 B+H vs B, loans only (n=2702): Intercept -0.0434 (se 0.0176, p 0.0134), log_true +0.0021 (se 0.0027, p 0.452), log_placebo -0.0075 (se 0.0023, p 0.00111), log_count +0.0060 (se 0.0025, p 0.0184)
- pythia-410m@step512 B+H vs B, exact counts (n=1744): Intercept -0.0759 (se 0.0244, p 0.00189), log_true -0.0037 (se 0.0047, p 0.43), log_placebo -0.0018 (se 0.0050, p 0.725), log_count +0.0141 (se 0.0046, p 0.00226)
- pythia-410m@step512 S+H vs S (n=3302): Intercept -0.2058 (se 0.0346, p 2.59e-09), log_true +0.0039 (se 0.0054, p 0.478), log_placebo -0.0066 (se 0.0048, p 0.166), log_count +0.0370 (se 0.0049, p 3.62e-14)
- pythia-410m@step512 S+H vs S, loans only (n=2702): Intercept -0.1602 (se 0.0363, p 1.01e-05), log_true +0.0051 (se 0.0065, p 0.432), log_placebo -0.0104 (se 0.0054, p 0.0527), log_count +0.0269 (se 0.0057, p 2.6e-06)
- pythia-410m@step512 S+H vs S, exact counts (n=1744): Intercept -0.1706 (se 0.0495, p 0.000571), log_true -0.0053 (se 0.0101, p 0.603), log_placebo -0.0029 (se 0.0114, p 0.802), log_count +0.0328 (se 0.0098, p 0.00082)
- pythia-410m@step8000 B+H vs B (n=3302): Intercept -0.0757 (se 0.0611, p 0.215), log_true +0.0261 (se 0.0100, p 0.00861), log_placebo -0.0256 (se 0.0092, p 0.00537), log_count +0.0218 (se 0.0095, p 0.0213)
- pythia-410m@step8000 B+H vs B, loans only (n=2702): Intercept -0.0568 (se 0.0666, p 0.394), log_true +0.0282 (se 0.0124, p 0.0225), log_placebo -0.0296 (se 0.0109, p 0.0067), log_count +0.0175 (se 0.0115, p 0.129)
- pythia-410m@step8000 B+H vs B, exact counts (n=1744): Intercept -0.0916 (se 0.0949, p 0.334), log_true +0.0140 (se 0.0195, p 0.474), log_placebo -0.0462 (se 0.0207, p 0.0254), log_count +0.0224 (se 0.0200, p 0.263)
- pythia-410m@step8000 S+H vs S (n=3302): Intercept -0.3605 (se 0.0961, p 0.000176), log_true +0.0405 (se 0.0150, p 0.00704), log_placebo -0.0229 (se 0.0143, p 0.11), log_count +0.0920 (se 0.0139, p 3.92e-11)
- pythia-410m@step8000 S+H vs S, loans only (n=2702): Intercept -0.2595 (se 0.1017, p 0.0108), log_true +0.0480 (se 0.0176, p 0.00629), log_placebo -0.0333 (se 0.0165, p 0.0428), log_count +0.0693 (se 0.0164, p 2.52e-05)
- pythia-410m@step8000 S+H vs S, exact counts (n=1744): Intercept -0.2506 (se 0.1414, p 0.0765), log_true +0.0386 (se 0.0283, p 0.173), log_placebo -0.0528 (se 0.0304, p 0.0827), log_count +0.0693 (se 0.0291, p 0.0173)
- pythia-70m@main B+H vs B (n=3302): Intercept -0.1165 (se 0.0447, p 0.00916), log_true +0.0045 (se 0.0071, p 0.527), log_placebo -0.0121 (se 0.0064, p 0.0587), log_count +0.0196 (se 0.0067, p 0.00353)
- pythia-70m@main B+H vs B, loans only (n=2702): Intercept -0.1379 (se 0.0486, p 0.00458), log_true +0.0025 (se 0.0089, p 0.778), log_placebo -0.0158 (se 0.0077, p 0.0394), log_count +0.0247 (se 0.0084, p 0.00309)
- pythia-70m@main B+H vs B, exact counts (n=1744): Intercept -0.0379 (se 0.0626, p 0.544), log_true +0.0014 (se 0.0133, p 0.917), log_placebo -0.0050 (se 0.0138, p 0.718), log_count +0.0026 (se 0.0134, p 0.847)
- pythia-70m@main S+H vs S (n=3302): Intercept -0.3937 (se 0.0772, p 3.35e-07), log_true +0.0079 (se 0.0121, p 0.515), log_placebo -0.0085 (se 0.0110, p 0.439), log_count +0.0763 (se 0.0110, p 4.26e-12)
- pythia-70m@main S+H vs S, loans only (n=2702): Intercept -0.3622 (se 0.0815, p 8.77e-06), log_true +0.0090 (se 0.0144, p 0.533), log_placebo -0.0150 (se 0.0128, p 0.242), log_count +0.0694 (se 0.0132, p 1.59e-07)
- pythia-70m@main S+H vs S, exact counts (n=1744): Intercept -0.1476 (se 0.1010, p 0.144), log_true +0.0079 (se 0.0217, p 0.716), log_placebo -0.0161 (se 0.0217, p 0.456), log_count +0.0274 (se 0.0213, p 0.197)
- words with counts: 3302 ; share with any true co-mention 77.3%; median true 7.0
- approx (subsampled) word counts: 47.2%
          median   mean  size
label                        
Arabic       2.0   10.2    65
Dutch        5.0   18.6   213
French      22.0  118.4   600
German       7.0   24.3   139
Greek        7.0   37.2   237
Indic        3.0   13.4   107
Italian      9.0   23.0   173
Japanese     9.0   73.4    81
Latin       15.0   92.9   600
Norse        1.0   21.0   296
Spanish      7.0   14.2   191
native       1.0   58.8   600

## Behaviour (asked directly, calibrated)
- pythia-1.4b@main: {'n': 2402, 'raw_bal_acc': 0.402, 'calibrated_bal_acc': 0.501, 'auc Norse vs native': 0.583, 'auc Dutch vs native': 0.698, 'probe_bal_acc_same_words': 0.581, 'per_class_calibrated': {'Arabic': 0.692, 'Dutch': 0.15, 'French': 0.153, 'German': 0.367, 'Greek': 0.802, 'Indic': 0.71, 'Italian': 0.653, 'Japanese': 0.741, 'Latin': 0.353, 'Norse': 0.392, 'Spanish': 0.733, 'native': 0.27}}
- pythia-2.8b@main: {'n': 2402, 'raw_bal_acc': 0.415, 'calibrated_bal_acc': 0.533, 'auc Norse vs native': 0.545, 'auc Dutch vs native': 0.771, 'probe_bal_acc_same_words': 0.578, 'per_class_calibrated': {'Arabic': 0.677, 'Dutch': 0.324, 'French': 0.113, 'German': 0.374, 'Greek': 0.84, 'Indic': 0.692, 'Italian': 0.671, 'Japanese': 0.827, 'Latin': 0.487, 'Norse': 0.429, 'Spanish': 0.775, 'native': 0.187}}
- pythia-410m@main: {'n': 2402, 'raw_bal_acc': 0.286, 'calibrated_bal_acc': 0.405, 'auc Norse vs native': 0.606, 'auc Dutch vs native': 0.72, 'probe_bal_acc_same_words': 0.542, 'per_class_calibrated': {'Arabic': 0.631, 'Dutch': 0.221, 'French': 0.143, 'German': 0.302, 'Greek': 0.489, 'Indic': 0.542, 'Italian': 0.561, 'Japanese': 0.691, 'Latin': 0.2, 'Norse': 0.287, 'Spanish': 0.518, 'native': 0.277}}
- pythia-70m@main: {'n': 2402, 'raw_bal_acc': 0.107, 'calibrated_bal_acc': 0.256, 'auc Norse vs native': 0.545, 'auc Dutch vs native': 0.558, 'probe_bal_acc_same_words': 0.422, 'per_class_calibrated': {'Arabic': 0.246, 'Dutch': 0.146, 'French': 0.117, 'German': 0.288, 'Greek': 0.291, 'Indic': 0.308, 'Italian': 0.295, 'Japanese': 0.568, 'Latin': 0.197, 'Norse': 0.193, 'Spanish': 0.335, 'native': 0.09}}
