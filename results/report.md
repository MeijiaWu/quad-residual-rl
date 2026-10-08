# Results

IQM over training runs with 95% bootstrap CI (Agarwal et al. 2021). Deltas and HF ratio are against `baseline_robust` on the same episodes (accuracy on episodes both survived). Fewer than 10 runs: indicative only.

## main

| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |
|---|---|---|---|---|---|---|
| main_residual | 10 | 0.000 [0.000, 0.000] | 0.253 [0.236, 0.274] | -0.059 [-0.076, -0.038] | 1.83 [1.68, 1.98] | 10/10 |
| main_direct | 10 | 0.000 [0.000, 0.002] | 0.111 [0.105, 0.123] | -0.201 [-0.206, -0.189] | 6.77 [6.44, 6.94] | 9/10 |

Baselines (trajectory=lemniscate, safety=off): `baseline_fast` crash 0.31, RMSE 0.147 m, HF 0.0798; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0058

## ablations

| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |
|---|---|---|---|---|---|---|
| abl_residual_fastbase | 10 | 0.098 [0.080, 0.125] | 0.147 [0.139, 0.157] | -0.165 [-0.173, -0.156] | 17.95 [17.17, 18.67] | 0/10 |
| abl_residual_scale05 | 10 | 0.000 [0.000, 0.000] | 0.265 [0.224, 0.299] | -0.047 [-0.088, -0.013] | 3.83 [3.50, 4.07] | 8/10 |
| abl_direct_nodelay | 10 | 0.000 [0.000, 0.005] | 0.116 [0.108, 0.122] | -0.196 [-0.203, -0.190] | 7.93 [7.24, 8.69] | 8/10 |
| abl_direct_norand | 10 | 0.002 [0.000, 0.008] | 0.212 [0.202, 0.233] | -0.100 [-0.109, -0.079] | 8.61 [7.88, 9.27] | 7/10 |
| abl_direct_nosmooth | 10 | 0.000 [0.000, 0.002] | 0.136 [0.123, 0.153] | -0.176 [-0.189, -0.159] | 7.20 [6.74, 7.65] | 9/10 |
| abl_direct_nocurriculum | 10 | 0.000 [0.000, 0.000] | 0.124 [0.118, 0.134] | -0.188 [-0.194, -0.178] | 6.82 [6.67, 7.26] | 10/10 |

Baselines (trajectory=lemniscate, safety=off): `baseline_fast` crash 0.31, RMSE 0.147 m, HF 0.0798; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0058

## generalization

| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |
|---|---|---|---|---|---|---|
| wp_residual | 10 | 0.002 [0.000, 0.007] | 0.593 [0.583, 0.605] | -0.121 [-0.131, -0.110] | 1.83 [1.71, 1.92] | 7/10 |
| wp_direct | 10 | 0.000 [0.000, 0.000] | 0.479 [0.466, 0.490] | -0.235 [-0.248, -0.224] | 7.82 [7.32, 7.98] | 10/10 |
| safety_residual | 10 | 0.000 [0.000, 0.000] | 0.254 [0.237, 0.275] | -0.058 [-0.075, -0.037] | 1.78 [1.65, 1.90] | 10/10 |
| safety_direct | 10 | 0.000 [0.000, 0.010] | 0.116 [0.111, 0.127] | -0.196 [-0.201, -0.185] | 6.28 [5.93, 6.43] | 8/10 |

Baselines (trajectory=waypoints, safety=off): `baseline_fast` crash 0.35, RMSE 0.534 m, HF 0.0906; `baseline_robust` crash 0.00, RMSE 0.714 m, HF 0.0061
Baselines (trajectory=lemniscate, safety=on): `baseline_fast` crash 0.95, RMSE 0.106 m, HF 0.0497; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0058

## stress

| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |
|---|---|---|---|---|---|---|
| stress_residual | 10 | 0.000 [0.000, 0.018] | 0.272 [0.253, 0.289] | -0.040 [-0.059, -0.024] | 1.96 [1.82, 2.16] | 7/10 |
| stress_direct | 10 | 0.157 [0.137, 0.220] | 0.126 [0.117, 0.139] | -0.185 [-0.193, -0.171] | 8.35 [7.80, 9.68] | 0/10 |
| stress_direct_safety | 10 | 0.182 [0.150, 0.243] | 0.189 [0.172, 0.211] | -0.120 [-0.137, -0.097] | 6.12 [5.73, 6.63] | 0/10 |
| stress_direct_nodelay | 10 | 0.300 [0.238, 0.355] | 0.159 [0.145, 0.171] | -0.152 [-0.166, -0.140] | 12.14 [10.22, 14.35] | 0/10 |

Baselines (trajectory=lemniscate, safety=off): `baseline_fast` crash 0.90, RMSE 0.270 m, HF 0.1227; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0057
Baselines (trajectory=lemniscate, safety=on): `baseline_fast` crash 1.00, RMSE nan m, HF nan; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0057

## supervisor

| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |
|---|---|---|---|---|---|---|
| sup_residual | 10 | 0.000 [0.000, 0.000] | 0.247 [0.229, 0.269] | -0.065 [-0.083, -0.042] | 1.67 [1.56, 1.79] | 10/10 |
| sup_direct | 10 | 0.000 [0.000, 0.000] | 0.125 [0.118, 0.138] | -0.187 [-0.194, -0.174] | 6.31 [5.93, 6.50] | 10/10 |
| stress_residual_sup | 10 | 0.000 [0.000, 0.002] | 0.270 [0.256, 0.286] | -0.043 [-0.056, -0.026] | 1.54 [1.45, 1.66] | 9/10 |
| stress_direct_sup | 10 | 0.017 [0.010, 0.045] | 0.192 [0.184, 0.201] | -0.121 [-0.129, -0.113] | 6.27 [5.75, 7.36] | 0/10 |

Baselines (trajectory=lemniscate, safety=off): `baseline_fast` crash 0.90, RMSE 0.270 m, HF 0.1227; `baseline_robust` crash 0.00, RMSE 0.312 m, HF 0.0057

