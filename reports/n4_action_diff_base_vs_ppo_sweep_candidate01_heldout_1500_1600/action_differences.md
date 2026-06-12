# Policy Action Difference Report

A: `base_best`
B: `ppo_sweep_candidate_01`
Episodes: `240`
Changed seeds: `114`
Action-changed seeds: `114`
Success gains B over A: `0`
Success losses B vs A: `0`

| model | mean | p10 | success | max |
|---|---:|---:|---:|---:|
| base_best | 484.5 | 441.9 | 0.692 | 500 |
| ppo_sweep_candidate_01 | 484.4 | 441.9 | 0.692 | 500 |

## Outcome Deltas

Mean delta steps (B - A): `-0.03`
First action diff median step: `225.0`
Success gain seeds: `[]`
Success loss seeds: `[]`

## Changed Seeds

| seed | A steps | B steps | delta | A failure | B failure | first diff | diff frac |
|---:|---:|---:|---:|---|---|---:|---:|
| 1500000 | 446 | 446 | 0 | pole_2_angle | pole_2_angle | 292 | 0.132 |
| 1500002 | 450 | 449 | -1 | pole_2_angle | pole_2_angle | 394 | 0.122 |
| 1500003 | 479 | 478 | -1 | pole_2_angle | pole_2_angle | 188 | 0.268 |
| 1500004 | 490 | 490 | 0 | pole_2_angle | pole_2_angle | 357 | 0.100 |
| 1500006 | 500 | 500 | 0 | success | success | 212 | 0.242 |
| 1500008 | 500 | 500 | 0 | success | success | 245 | 0.122 |
| 1500011 | 432 | 432 | 0 | pole_2_angle | pole_2_angle | 266 | 0.171 |
| 1500012 | 500 | 500 | 0 | success | success | 21 | 0.374 |
| 1500014 | 500 | 500 | 0 | success | success | 178 | 0.292 |
| 1500021 | 500 | 500 | 0 | success | success | 81 | 0.402 |
| 1500022 | 500 | 500 | 0 | success | success | 0 | 0.498 |
| 1500023 | 479 | 478 | -1 | pole_2_angle | pole_2_angle | 426 | 0.052 |
| 1500025 | 500 | 500 | 0 | success | success | 266 | 0.180 |
| 1500031 | 500 | 500 | 0 | success | success | 125 | 0.272 |
| 1500032 | 455 | 454 | -1 | pole_2_angle | pole_2_angle | 293 | 0.123 |
| 1500033 | 500 | 500 | 0 | success | success | 133 | 0.322 |
| 1500035 | 447 | 447 | 0 | pole_2_angle | pole_2_angle | 286 | 0.148 |
| 1500036 | 390 | 390 | 0 | pole_1_angle | pole_1_angle | 339 | 0.046 |
| 1500037 | 446 | 446 | 0 | pole_2_angle | pole_2_angle | 375 | 0.045 |
| 1500043 | 500 | 500 | 0 | success | success | 151 | 0.298 |
| 1500045 | 468 | 468 | 0 | pole_0_angle | pole_0_angle | 288 | 0.186 |
| 1500047 | 442 | 442 | 0 | pole_1_angle | pole_1_angle | 418 | 0.041 |
| 1500048 | 500 | 500 | 0 | success | success | 112 | 0.198 |
| 1500053 | 500 | 500 | 0 | success | success | 46 | 0.352 |
| 1500055 | 500 | 500 | 0 | success | success | 0 | 0.558 |
| 1500056 | 500 | 500 | 0 | success | success | 257 | 0.228 |
| 1500057 | 500 | 500 | 0 | success | success | 296 | 0.030 |
| 1500060 | 500 | 500 | 0 | success | success | 483 | 0.016 |
| 1500061 | 418 | 418 | 0 | pole_2_angle | pole_2_angle | 308 | 0.077 |
| 1500063 | 417 | 416 | -1 | pole_2_angle | pole_2_angle | 350 | 0.159 |
| 1500065 | 500 | 500 | 0 | success | success | 4 | 0.466 |
| 1500067 | 500 | 500 | 0 | success | success | 236 | 0.268 |
| 1500072 | 500 | 500 | 0 | success | success | 107 | 0.194 |
| 1500073 | 461 | 461 | 0 | pole_2_angle | pole_2_angle | 410 | 0.039 |
| 1500074 | 500 | 500 | 0 | success | success | 356 | 0.150 |
| 1500076 | 500 | 500 | 0 | success | success | 25 | 0.418 |
| 1500077 | 500 | 500 | 0 | success | success | 0 | 0.494 |
| 1500079 | 500 | 500 | 0 | success | success | 419 | 0.080 |
| 1500082 | 500 | 500 | 0 | success | success | 370 | 0.078 |
| 1500085 | 500 | 500 | 0 | success | success | 351 | 0.144 |
| 1500093 | 465 | 468 | 3 | pole_2_angle | pole_2_angle | 220 | 0.103 |
| 1500095 | 500 | 500 | 0 | success | success | 94 | 0.232 |
| 1500096 | 500 | 500 | 0 | success | success | 123 | 0.330 |
| 1500099 | 500 | 500 | 0 | success | success | 148 | 0.294 |
| 1500102 | 464 | 463 | -1 | pole_2_angle | pole_2_angle | 259 | 0.175 |
| 1500103 | 500 | 500 | 0 | success | success | 78 | 0.400 |
| 1500106 | 500 | 500 | 0 | success | success | 479 | 0.010 |
| 1500110 | 489 | 488 | -1 | pole_2_angle | pole_2_angle | 443 | 0.031 |
| 1500114 | 500 | 500 | 0 | success | success | 266 | 0.168 |
| 1500115 | 500 | 500 | 0 | success | success | 0 | 0.376 |
| 1500117 | 397 | 397 | 0 | pole_1_angle | pole_1_angle | 324 | 0.038 |
| 1500119 | 500 | 500 | 0 | success | success | 112 | 0.256 |
| 1600000 | 500 | 500 | 0 | success | success | 220 | 0.214 |
| 1600001 | 500 | 500 | 0 | success | success | 455 | 0.034 |
| 1600004 | 500 | 500 | 0 | success | success | 0 | 0.608 |
| 1600006 | 500 | 500 | 0 | success | success | 273 | 0.162 |
| 1600007 | 500 | 500 | 0 | success | success | 15 | 0.476 |
| 1600008 | 500 | 500 | 0 | success | success | 137 | 0.236 |
| 1600011 | 500 | 500 | 0 | success | success | 480 | 0.040 |
| 1600012 | 500 | 500 | 0 | success | success | 201 | 0.258 |
| 1600013 | 500 | 500 | 0 | success | success | 324 | 0.050 |
| 1600014 | 500 | 500 | 0 | success | success | 151 | 0.372 |
| 1600016 | 500 | 500 | 0 | success | success | 177 | 0.278 |
| 1600023 | 500 | 500 | 0 | success | success | 20 | 0.442 |
| 1600026 | 500 | 500 | 0 | success | success | 3 | 0.532 |
| 1600031 | 500 | 500 | 0 | success | success | 15 | 0.424 |
| 1600037 | 500 | 500 | 0 | success | success | 327 | 0.114 |
| 1600040 | 500 | 500 | 0 | success | success | 170 | 0.080 |
| 1600042 | 500 | 500 | 0 | success | success | 126 | 0.400 |
| 1600049 | 500 | 500 | 0 | success | success | 390 | 0.112 |
| 1600050 | 500 | 500 | 0 | success | success | 139 | 0.426 |
| 1600051 | 500 | 500 | 0 | success | success | 376 | 0.090 |
| 1600056 | 500 | 500 | 0 | success | success | 116 | 0.322 |
| 1600057 | 500 | 500 | 0 | success | success | 33 | 0.422 |
| 1600058 | 500 | 500 | 0 | success | success | 323 | 0.154 |
| 1600060 | 500 | 500 | 0 | success | success | 239 | 0.022 |
| 1600062 | 424 | 422 | -2 | pole_2_angle | pole_2_angle | 332 | 0.213 |
| 1600063 | 500 | 500 | 0 | success | success | 65 | 0.424 |
| 1600064 | 500 | 500 | 0 | success | success | 339 | 0.140 |
| 1600066 | 500 | 500 | 0 | success | success | 205 | 0.228 |
| 1600067 | 500 | 500 | 0 | success | success | 270 | 0.068 |
| 1600069 | 500 | 500 | 0 | success | success | 276 | 0.062 |
| 1600070 | 500 | 500 | 0 | success | success | 20 | 0.420 |
| 1600073 | 473 | 473 | 0 | pole_2_angle | pole_2_angle | 370 | 0.070 |
| 1600074 | 500 | 500 | 0 | success | success | 14 | 0.274 |
| 1600076 | 500 | 500 | 0 | success | success | 195 | 0.144 |
| 1600078 | 500 | 500 | 0 | success | success | 488 | 0.024 |
| 1600079 | 418 | 418 | 0 | pole_1_angle | pole_1_angle | 382 | 0.043 |
| 1600080 | 500 | 500 | 0 | success | success | 140 | 0.316 |
| 1600081 | 500 | 500 | 0 | success | success | 165 | 0.240 |
| 1600083 | 500 | 500 | 0 | success | success | 495 | 0.006 |
| 1600084 | 442 | 442 | 0 | pole_2_angle | pole_2_angle | 291 | 0.014 |
| 1600085 | 500 | 500 | 0 | success | success | 14 | 0.410 |
| 1600086 | 500 | 500 | 0 | success | success | 0 | 0.512 |
| 1600088 | 500 | 500 | 0 | success | success | 48 | 0.418 |
| 1600089 | 500 | 500 | 0 | success | success | 87 | 0.410 |
| 1600090 | 500 | 500 | 0 | success | success | 184 | 0.294 |
| 1600091 | 500 | 500 | 0 | success | success | 235 | 0.270 |
| 1600093 | 500 | 500 | 0 | success | success | 131 | 0.364 |
| 1600095 | 463 | 463 | 0 | pole_2_angle | pole_2_angle | 315 | 0.132 |
| 1600097 | 500 | 500 | 0 | success | success | 42 | 0.376 |
| 1600098 | 500 | 500 | 0 | success | success | 90 | 0.390 |
| 1600101 | 474 | 474 | 0 | pole_2_angle | pole_2_angle | 357 | 0.082 |
| 1600102 | 500 | 500 | 0 | success | success | 386 | 0.034 |
| 1600104 | 485 | 484 | -1 | pole_2_angle | pole_2_angle | 412 | 0.149 |
| 1600108 | 443 | 442 | -1 | pole_2_angle | pole_2_angle | 351 | 0.206 |
| 1600109 | 500 | 500 | 0 | success | success | 95 | 0.388 |
| 1600110 | 500 | 500 | 0 | success | success | 74 | 0.408 |
| 1600111 | 500 | 500 | 0 | success | success | 230 | 0.162 |
| 1600112 | 500 | 500 | 0 | success | success | 1 | 0.454 |
| 1600113 | 500 | 500 | 0 | success | success | 0 | 0.528 |
| 1600116 | 500 | 500 | 0 | success | success | 438 | 0.016 |
| 1600118 | 492 | 492 | 0 | pole_2_angle | pole_2_angle | 361 | 0.041 |
| 1600119 | 500 | 500 | 0 | success | success | 323 | 0.162 |

## First-Difference Examples

### Seed 1500093

Steps: `465` -> `468`; first diff step `220`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1600062

Steps: `424` -> `422`; first diff step `332`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500002

Steps: `450` -> `449`; first diff step `394`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500003

Steps: `479` -> `478`; first diff step `188`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500023

Steps: `479` -> `478`; first diff step `426`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500032

Steps: `455` -> `454`; first diff step `293`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500063

Steps: `417` -> `416`; first diff step `350`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500102

Steps: `464` -> `463`; first diff step `259`.
Actions: `0` -> `4`; forces `-10.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500110

Steps: `489` -> `488`; first diff step `443`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1600104

Steps: `485` -> `484`; first diff step `412`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1600108

Steps: `443` -> `442`; first diff step `351`.
Actions: `4` -> `2`; forces `10.0` -> `0.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

### Seed 1500000

Steps: `446` -> `446`; first diff step `292`.
Actions: `1` -> `4`; forces `-5.0` -> `10.0`.
Regimes: `stabilize_chain` -> `stabilize_chain`.

