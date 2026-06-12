# Policy Action Difference Report

A: `base_best`
B: `targetkl_chunk1`
Episodes: `240`
Changed seeds: `114`
Action-changed seeds: `114`
Success gains B over A: `1`
Success losses B vs A: `0`

| model | mean | p10 | success | max |
|---|---:|---:|---:|---:|
| base_best | 484.9 | 434.9 | 0.696 | 500 |
| targetkl_chunk1 | 484.9 | 434.9 | 0.700 | 500 |

## Outcome Deltas

Mean delta steps (B - A): `0.01`
First action diff median step: `240.0`
Success gain seeds: `[930013]`
Success loss seeds: `[]`

## Changed Seeds

| seed | A steps | B steps | delta | A failure | B failure | first diff | diff frac |
|---:|---:|---:|---:|---|---|---:|---:|
| 900000 | 500 | 500 | 0 | success | success | 428 | 0.064 |
| 900001 | 500 | 500 | 0 | success | success | 410 | 0.114 |
| 900004 | 500 | 500 | 0 | success | success | 81 | 0.464 |
| 900006 | 500 | 500 | 0 | success | success | 435 | 0.060 |
| 900007 | 500 | 500 | 0 | success | success | 254 | 0.230 |
| 900009 | 500 | 500 | 0 | success | success | 444 | 0.050 |
| 900010 | 500 | 500 | 0 | success | success | 358 | 0.146 |
| 900011 | 500 | 500 | 0 | success | success | 102 | 0.418 |
| 900015 | 500 | 500 | 0 | success | success | 259 | 0.108 |
| 900016 | 500 | 500 | 0 | success | success | 127 | 0.332 |
| 900017 | 500 | 500 | 0 | success | success | 487 | 0.026 |
| 900019 | 500 | 500 | 0 | success | success | 48 | 0.416 |
| 900022 | 500 | 500 | 0 | success | success | 28 | 0.378 |
| 900024 | 500 | 500 | 0 | success | success | 418 | 0.072 |
| 900026 | 500 | 500 | 0 | success | success | 339 | 0.170 |
| 930000 | 500 | 500 | 0 | success | success | 0 | 0.552 |
| 930001 | 500 | 500 | 0 | success | success | 484 | 0.028 |
| 930002 | 500 | 500 | 0 | success | success | 400 | 0.028 |
| 930004 | 466 | 466 | 0 | pole_1_angle | pole_1_angle | 437 | 0.028 |
| 930011 | 500 | 500 | 0 | success | success | 13 | 0.464 |
| 930012 | 500 | 500 | 0 | success | success | 84 | 0.372 |
| 930013 | 488 | 500 | 12 | pole_1_angle | success | 0 | 0.398 |
| 930015 | 500 | 500 | 0 | success | success | 0 | 0.188 |
| 930016 | 500 | 500 | 0 | success | success | 12 | 0.440 |
| 930019 | 500 | 500 | 0 | success | success | 328 | 0.218 |
| 930020 | 460 | 462 | 2 | pole_2_angle | pole_2_angle | 310 | 0.150 |
| 930021 | 489 | 489 | 0 | pole_0_angle | pole_0_angle | 324 | 0.157 |
| 930025 | 500 | 500 | 0 | success | success | 11 | 0.092 |
| 970000 | 500 | 500 | 0 | success | success | 238 | 0.132 |
| 970001 | 500 | 500 | 0 | success | success | 460 | 0.054 |
| 970003 | 500 | 500 | 0 | success | success | 449 | 0.064 |
| 970004 | 500 | 500 | 0 | success | success | 242 | 0.042 |
| 970008 | 500 | 500 | 0 | success | success | 294 | 0.192 |
| 970009 | 500 | 500 | 0 | success | success | 436 | 0.052 |
| 970011 | 500 | 500 | 0 | success | success | 0 | 0.468 |
| 970014 | 500 | 500 | 0 | success | success | 27 | 0.354 |
| 970018 | 453 | 452 | -1 | pole_0_angle | pole_2_angle | 370 | 0.093 |
| 970023 | 500 | 500 | 0 | success | success | 427 | 0.140 |
| 970027 | 500 | 500 | 0 | success | success | 255 | 0.224 |
| 970028 | 500 | 500 | 0 | success | success | 25 | 0.518 |
| 970029 | 461 | 461 | 0 | pole_2_angle | pole_2_angle | 347 | 0.108 |
| 1010000 | 500 | 500 | 0 | success | success | 226 | 0.182 |
| 1010004 | 500 | 500 | 0 | success | success | 344 | 0.044 |
| 1010005 | 500 | 500 | 0 | success | success | 0 | 0.442 |
| 1010008 | 500 | 500 | 0 | success | success | 143 | 0.370 |
| 1010009 | 500 | 500 | 0 | success | success | 416 | 0.046 |
| 1010010 | 500 | 500 | 0 | success | success | 186 | 0.252 |
| 1010012 | 500 | 500 | 0 | success | success | 332 | 0.180 |
| 1010016 | 500 | 500 | 0 | success | success | 226 | 0.280 |
| 1010018 | 500 | 500 | 0 | success | success | 463 | 0.074 |
| 1010022 | 440 | 440 | 0 | pole_2_angle | pole_2_angle | 312 | 0.111 |
| 1010023 | 500 | 500 | 0 | success | success | 0 | 0.504 |
| 1010024 | 432 | 433 | 1 | pole_1_angle | pole_1_angle | 370 | 0.144 |
| 1010027 | 500 | 500 | 0 | success | success | 101 | 0.406 |
| 1010028 | 500 | 500 | 0 | success | success | 0 | 0.500 |
| 1010029 | 480 | 480 | 0 | pole_2_angle | pole_2_angle | 269 | 0.244 |
| 1040002 | 469 | 468 | -1 | pole_2_angle | pole_2_angle | 411 | 0.060 |
| 1040004 | 500 | 500 | 0 | success | success | 396 | 0.038 |
| 1040005 | 500 | 500 | 0 | success | success | 232 | 0.208 |
| 1040008 | 430 | 427 | -3 | pole_2_angle | pole_2_angle | 325 | 0.115 |
| 1040010 | 500 | 500 | 0 | success | success | 259 | 0.046 |
| 1040011 | 500 | 500 | 0 | success | success | 149 | 0.278 |
| 1040013 | 500 | 500 | 0 | success | success | 193 | 0.246 |
| 1040014 | 500 | 500 | 0 | success | success | 7 | 0.494 |
| 1040017 | 476 | 473 | -3 | pole_2_angle | pole_2_angle | 61 | 0.378 |
| 1040020 | 500 | 500 | 0 | success | success | 171 | 0.176 |
| 1040021 | 500 | 500 | 0 | success | success | 0 | 0.516 |
| 1040023 | 405 | 405 | 0 | pole_2_angle | pole_2_angle | 230 | 0.077 |
| 1040024 | 493 | 493 | 0 | pole_1_angle | pole_1_angle | 447 | 0.091 |
| 1070000 | 469 | 470 | 1 | pole_2_angle | pole_2_angle | 384 | 0.177 |
| 1070002 | 443 | 443 | 0 | pole_2_angle | pole_2_angle | 334 | 0.102 |
| 1070003 | 500 | 500 | 0 | success | success | 9 | 0.488 |
| 1070004 | 496 | 493 | -3 | pole_2_angle | pole_2_angle | 386 | 0.112 |
| 1070012 | 500 | 500 | 0 | success | success | 93 | 0.418 |
| 1070013 | 500 | 500 | 0 | success | success | 86 | 0.432 |
| 1070014 | 500 | 500 | 0 | success | success | 335 | 0.092 |
| 1070015 | 500 | 500 | 0 | success | success | 44 | 0.408 |
| 1070017 | 500 | 500 | 0 | success | success | 111 | 0.360 |
| 1070018 | 486 | 486 | 0 | pole_2_angle | pole_2_angle | 325 | 0.056 |
| 1070021 | 446 | 444 | -2 | pole_2_angle | pole_2_angle | 380 | 0.068 |
| 1070022 | 500 | 500 | 0 | success | success | 416 | 0.074 |
| 1070023 | 500 | 500 | 0 | success | success | 431 | 0.032 |
| 1070024 | 500 | 500 | 0 | success | success | 114 | 0.280 |
| 1070028 | 500 | 500 | 0 | success | success | 0 | 0.520 |
| 1140001 | 421 | 421 | 0 | pole_1_angle | pole_1_angle | 382 | 0.090 |
| 1140002 | 494 | 494 | 0 | pole_2_angle | pole_2_angle | 378 | 0.095 |
| 1140004 | 489 | 489 | 0 | pole_0_angle | pole_0_angle | 330 | 0.129 |
| 1140006 | 500 | 500 | 0 | success | success | 0 | 0.472 |
| 1140007 | 500 | 500 | 0 | success | success | 220 | 0.198 |
| 1140008 | 500 | 500 | 0 | success | success | 397 | 0.056 |
| 1140009 | 435 | 435 | 0 | pole_2_angle | pole_2_angle | 254 | 0.032 |
| 1140010 | 445 | 445 | 0 | pole_0_angle | pole_0_angle | 276 | 0.081 |
| 1140014 | 500 | 500 | 0 | success | success | 378 | 0.106 |
| 1140016 | 500 | 500 | 0 | success | success | 233 | 0.246 |
| 1140017 | 500 | 500 | 0 | success | success | 14 | 0.528 |
| 1140018 | 383 | 383 | 0 | pole_1_angle | pole_1_angle | 370 | 0.013 |
| 1140019 | 500 | 500 | 0 | success | success | 207 | 0.296 |
| 1140020 | 500 | 500 | 0 | success | success | 113 | 0.310 |
| 1140021 | 500 | 500 | 0 | success | success | 71 | 0.416 |
| 1140023 | 500 | 500 | 0 | success | success | 213 | 0.250 |
| 1140024 | 500 | 500 | 0 | success | success | 202 | 0.230 |
| 1140027 | 447 | 447 | 0 | pole_2_angle | pole_2_angle | 306 | 0.134 |
| 1140028 | 500 | 500 | 0 | success | success | 106 | 0.256 |
| 1300002 | 500 | 500 | 0 | success | success | 353 | 0.222 |
| 1300004 | 435 | 435 | 0 | pole_2_angle | pole_2_angle | 370 | 0.080 |
| 1300008 | 500 | 500 | 0 | success | success | 382 | 0.092 |
| 1300011 | 500 | 500 | 0 | success | success | 22 | 0.566 |
| 1300013 | 500 | 500 | 0 | success | success | 207 | 0.260 |
| 1300015 | 500 | 500 | 0 | success | success | 0 | 0.494 |
| 1300021 | 500 | 500 | 0 | success | success | 177 | 0.350 |
| 1300022 | 500 | 500 | 0 | success | success | 104 | 0.314 |
| 1300027 | 500 | 500 | 0 | success | success | 107 | 0.254 |
| 1300028 | 500 | 500 | 0 | success | success | 183 | 0.314 |
| 1300029 | 500 | 500 | 0 | success | success | 0 | 0.424 |

## First-Difference Examples

### Seed 930013

Steps: `488` -> `500`; first diff step `0`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1040008

Steps: `430` -> `427`; first diff step `325`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1040017

Steps: `476` -> `473`; first diff step `61`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1070004

Steps: `496` -> `493`; first diff step `386`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 930020

Steps: `460` -> `462`; first diff step `310`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1070021

Steps: `446` -> `444`; first diff step `380`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 970018

Steps: `453` -> `452`; first diff step `370`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1010024

Steps: `432` -> `433`; first diff step `370`.
Actions: `4` -> `3`; forces `10.0` -> `5.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1040002

Steps: `469` -> `468`; first diff step `411`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1070000

Steps: `469` -> `470`; first diff step `384`.
Actions: `4` -> `3`; forces `10.0` -> `5.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900000

Steps: `500` -> `500`; first diff step `428`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900001

Steps: `500` -> `500`; first diff step `410`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

