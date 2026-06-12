# Policy Action Difference Report

A: `base_best`
B: `lexicographic_015k`
Episodes: `240`
Changed seeds: `107`
Action-changed seeds: `107`
Success gains B over A: `1`
Success losses B vs A: `0`

| model | mean | p10 | success | max |
|---|---:|---:|---:|---:|
| base_best | 484.9 | 434.9 | 0.696 | 500 |
| lexicographic_015k | 484.9 | 434.9 | 0.700 | 500 |

## Outcome Deltas

Mean delta steps (B - A): `0.04`
First action diff median step: `254.0`
Success gain seeds: `[930013]`
Success loss seeds: `[]`

## Changed Seeds

| seed | A steps | B steps | delta | A failure | B failure | first diff | diff frac |
|---:|---:|---:|---:|---|---|---:|---:|
| 900001 | 500 | 500 | 0 | success | success | 410 | 0.180 |
| 900004 | 500 | 500 | 0 | success | success | 81 | 0.460 |
| 900006 | 500 | 500 | 0 | success | success | 456 | 0.082 |
| 900007 | 500 | 500 | 0 | success | success | 254 | 0.220 |
| 900009 | 500 | 500 | 0 | success | success | 467 | 0.064 |
| 900010 | 500 | 500 | 0 | success | success | 392 | 0.100 |
| 900011 | 500 | 500 | 0 | success | success | 102 | 0.406 |
| 900015 | 500 | 500 | 0 | success | success | 259 | 0.412 |
| 900016 | 500 | 500 | 0 | success | success | 57 | 0.442 |
| 900017 | 500 | 500 | 0 | success | success | 492 | 0.016 |
| 900019 | 500 | 500 | 0 | success | success | 0 | 0.508 |
| 900022 | 500 | 500 | 0 | success | success | 47 | 0.388 |
| 900024 | 500 | 500 | 0 | success | success | 493 | 0.008 |
| 900026 | 500 | 500 | 0 | success | success | 281 | 0.252 |
| 930000 | 500 | 500 | 0 | success | success | 0 | 0.476 |
| 930002 | 500 | 500 | 0 | success | success | 414 | 0.136 |
| 930004 | 466 | 466 | 0 | pole_1_angle | pole_1_angle | 450 | 0.034 |
| 930011 | 500 | 500 | 0 | success | success | 13 | 0.464 |
| 930012 | 500 | 500 | 0 | success | success | 84 | 0.342 |
| 930013 | 488 | 500 | 12 | pole_1_angle | success | 0 | 0.406 |
| 930015 | 500 | 500 | 0 | success | success | 0 | 0.244 |
| 930016 | 500 | 500 | 0 | success | success | 11 | 0.406 |
| 930019 | 500 | 500 | 0 | success | success | 328 | 0.100 |
| 930020 | 460 | 459 | -1 | pole_2_angle | pole_2_angle | 310 | 0.316 |
| 930021 | 489 | 487 | -2 | pole_0_angle | pole_2_angle | 336 | 0.296 |
| 930025 | 500 | 500 | 0 | success | success | 11 | 0.092 |
| 970000 | 500 | 500 | 0 | success | success | 271 | 0.058 |
| 970001 | 500 | 500 | 0 | success | success | 484 | 0.032 |
| 970003 | 500 | 500 | 0 | success | success | 452 | 0.058 |
| 970004 | 500 | 500 | 0 | success | success | 242 | 0.134 |
| 970008 | 500 | 500 | 0 | success | success | 294 | 0.220 |
| 970009 | 500 | 500 | 0 | success | success | 447 | 0.096 |
| 970011 | 500 | 500 | 0 | success | success | 0 | 0.490 |
| 970014 | 500 | 500 | 0 | success | success | 0 | 0.498 |
| 970023 | 500 | 500 | 0 | success | success | 458 | 0.084 |
| 970027 | 500 | 500 | 0 | success | success | 436 | 0.060 |
| 970028 | 500 | 500 | 0 | success | success | 25 | 0.536 |
| 970029 | 461 | 461 | 0 | pole_2_angle | pole_2_angle | 372 | 0.184 |
| 1010000 | 500 | 500 | 0 | success | success | 262 | 0.124 |
| 1010004 | 500 | 500 | 0 | success | success | 384 | 0.008 |
| 1010005 | 500 | 500 | 0 | success | success | 478 | 0.020 |
| 1010008 | 500 | 500 | 0 | success | success | 92 | 0.478 |
| 1010009 | 500 | 500 | 0 | success | success | 438 | 0.116 |
| 1010010 | 500 | 500 | 0 | success | success | 226 | 0.194 |
| 1010012 | 500 | 500 | 0 | success | success | 254 | 0.222 |
| 1010016 | 500 | 500 | 0 | success | success | 405 | 0.066 |
| 1010018 | 500 | 500 | 0 | success | success | 482 | 0.036 |
| 1010022 | 440 | 440 | 0 | pole_2_angle | pole_2_angle | 328 | 0.230 |
| 1010023 | 500 | 500 | 0 | success | success | 0 | 0.508 |
| 1010024 | 432 | 433 | 1 | pole_1_angle | pole_1_angle | 385 | 0.109 |
| 1010027 | 500 | 500 | 0 | success | success | 3 | 0.496 |
| 1010028 | 500 | 500 | 0 | success | success | 0 | 0.468 |
| 1010029 | 480 | 480 | 0 | pole_2_angle | pole_2_angle | 269 | 0.210 |
| 1040004 | 500 | 500 | 0 | success | success | 442 | 0.010 |
| 1040005 | 500 | 500 | 0 | success | success | 232 | 0.190 |
| 1040010 | 500 | 500 | 0 | success | success | 259 | 0.144 |
| 1040011 | 500 | 500 | 0 | success | success | 170 | 0.462 |
| 1040013 | 500 | 500 | 0 | success | success | 140 | 0.334 |
| 1040014 | 500 | 500 | 0 | success | success | 7 | 0.488 |
| 1040017 | 476 | 476 | 0 | pole_2_angle | pole_2_angle | 223 | 0.118 |
| 1040020 | 500 | 500 | 0 | success | success | 191 | 0.160 |
| 1040021 | 500 | 500 | 0 | success | success | 0 | 0.516 |
| 1040023 | 405 | 405 | 0 | pole_2_angle | pole_2_angle | 249 | 0.168 |
| 1040024 | 493 | 493 | 0 | pole_1_angle | pole_1_angle | 456 | 0.075 |
| 1070000 | 469 | 470 | 1 | pole_2_angle | pole_2_angle | 406 | 0.134 |
| 1070002 | 443 | 443 | 0 | pole_2_angle | pole_2_angle | 345 | 0.212 |
| 1070003 | 500 | 500 | 0 | success | success | 9 | 0.478 |
| 1070012 | 500 | 500 | 0 | success | success | 0 | 0.504 |
| 1070013 | 500 | 500 | 0 | success | success | 67 | 0.412 |
| 1070014 | 500 | 500 | 0 | success | success | 407 | 0.018 |
| 1070015 | 500 | 500 | 0 | success | success | 14 | 0.432 |
| 1070017 | 500 | 500 | 0 | success | success | 111 | 0.364 |
| 1070018 | 486 | 486 | 0 | pole_2_angle | pole_2_angle | 340 | 0.183 |
| 1070022 | 500 | 500 | 0 | success | success | 433 | 0.128 |
| 1070023 | 500 | 500 | 0 | success | success | 448 | 0.100 |
| 1070024 | 500 | 500 | 0 | success | success | 65 | 0.364 |
| 1070028 | 500 | 500 | 0 | success | success | 0 | 0.482 |
| 1140001 | 421 | 421 | 0 | pole_1_angle | pole_1_angle | 386 | 0.083 |
| 1140002 | 494 | 494 | 0 | pole_2_angle | pole_2_angle | 402 | 0.172 |
| 1140004 | 489 | 488 | -1 | pole_0_angle | pole_0_angle | 345 | 0.176 |
| 1140006 | 500 | 500 | 0 | success | success | 0 | 0.472 |
| 1140007 | 500 | 500 | 0 | success | success | 309 | 0.042 |
| 1140008 | 500 | 500 | 0 | success | success | 465 | 0.028 |
| 1140009 | 435 | 435 | 0 | pole_2_angle | pole_2_angle | 269 | 0.221 |
| 1140010 | 445 | 445 | 0 | pole_0_angle | pole_0_angle | 294 | 0.153 |
| 1140014 | 500 | 500 | 0 | success | success | 399 | 0.148 |
| 1140016 | 500 | 500 | 0 | success | success | 233 | 0.230 |
| 1140017 | 500 | 500 | 0 | success | success | 14 | 0.450 |
| 1140018 | 383 | 383 | 0 | pole_1_angle | pole_1_angle | 375 | 0.021 |
| 1140019 | 500 | 500 | 0 | success | success | 129 | 0.312 |
| 1140020 | 500 | 500 | 0 | success | success | 104 | 0.342 |
| 1140021 | 500 | 500 | 0 | success | success | 25 | 0.520 |
| 1140023 | 500 | 500 | 0 | success | success | 193 | 0.250 |
| 1140024 | 500 | 500 | 0 | success | success | 243 | 0.294 |
| 1140027 | 447 | 447 | 0 | pole_2_angle | pole_2_angle | 322 | 0.190 |
| 1140028 | 500 | 500 | 0 | success | success | 209 | 0.098 |
| 1300002 | 500 | 500 | 0 | success | success | 353 | 0.098 |
| 1300004 | 435 | 435 | 0 | pole_2_angle | pole_2_angle | 394 | 0.090 |
| 1300008 | 500 | 500 | 0 | success | success | 402 | 0.184 |
| 1300011 | 500 | 500 | 0 | success | success | 0 | 0.626 |
| 1300013 | 500 | 500 | 0 | success | success | 153 | 0.332 |
| 1300015 | 500 | 500 | 0 | success | success | 0 | 0.514 |
| 1300021 | 500 | 500 | 0 | success | success | 177 | 0.366 |
| 1300022 | 500 | 500 | 0 | success | success | 51 | 0.444 |
| 1300027 | 500 | 500 | 0 | success | success | 3 | 0.466 |
| 1300028 | 500 | 500 | 0 | success | success | 183 | 0.330 |
| 1300029 | 500 | 500 | 0 | success | success | 0 | 0.482 |

## First-Difference Examples

### Seed 930013

Steps: `488` -> `500`; first diff step `0`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 930021

Steps: `489` -> `487`; first diff step `336`.
Actions: `3` -> `4`; forces `5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 930020

Steps: `460` -> `459`; first diff step `310`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1010024

Steps: `432` -> `433`; first diff step `385`.
Actions: `2` -> `4`; forces `0.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1070000

Steps: `469` -> `470`; first diff step `406`.
Actions: `2` -> `4`; forces `0.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1140004

Steps: `489` -> `488`; first diff step `345`.
Actions: `3` -> `4`; forces `5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900001

Steps: `500` -> `500`; first diff step `410`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900004

Steps: `500` -> `500`; first diff step `81`.
Actions: `4` -> `0`; forces `10.0` -> `-10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900006

Steps: `500` -> `500`; first diff step `456`.
Actions: `3` -> `4`; forces `5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900007

Steps: `500` -> `500`; first diff step `254`.
Actions: `4` -> `0`; forces `10.0` -> `-10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900009

Steps: `500` -> `500`; first diff step `467`.
Actions: `3` -> `4`; forces `5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 900010

Steps: `500` -> `500`; first diff step `392`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

