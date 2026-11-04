## Pooled quiz drops on gated papers (orig minus version; bootstrap over questions within papers)

| reader | statistic | mean [95% CI] | papers with CI above 0 | papers with mean above 0 |
|---|---|---|---|---|
| gpt-5.6-sol | drop_rem | +0.396 [+0.366, +0.424] | 11 of 11 | 11 of 11 |
| gpt-5.6-sol | drop_imp | +0.005 [-0.012, +0.023] | 0 of 6 | 5 of 6 |
| gemma-4-31b-it | drop_rem | +0.288 [+0.262, +0.315] | 11 of 11 | 11 of 11 |
| gemma-4-31b-it | drop_rem_minus_smart | -0.107 [-0.135, -0.079] | 0 of 11 | 1 of 11 |
| gemma-4-31b-it | drop_imp | -0.005 [-0.025, +0.015] | 0 of 6 | 2 of 6 |
| gemma-4-31b-it | drop_imp_minus_smart | -0.010 [-0.036, +0.017] | 0 of 6 | 3 of 6 |
| ministral-8b-2512 | drop_rem | +0.109 [+0.091, +0.128] | 10 of 11 | 11 of 11 |
| ministral-8b-2512 | drop_rem_minus_smart | -0.286 [-0.317, -0.255] | 0 of 11 | 0 of 11 |
| ministral-8b-2512 | drop_imp | +0.021 [+0.002, +0.039] | 1 of 6 | 5 of 6 |
| ministral-8b-2512 | drop_imp_minus_smart | +0.016 [-0.009, +0.040] | 1 of 6 | 4 of 6 |
| llama-3.2-3b-instruct | drop_rem | +0.014 [+0.003, +0.025] | 2 of 11 | 9 of 11 |
| llama-3.2-3b-instruct | drop_rem_minus_smart | -0.382 [-0.411, -0.351] | 0 of 11 | 0 of 11 |
| llama-3.2-3b-instruct | drop_imp | +0.003 [-0.013, +0.019] | 0 of 6 | 2 of 6 |
| llama-3.2-3b-instruct | drop_imp_minus_smart | -0.002 [-0.025, +0.021] | 0 of 6 | 3 of 6 |

Information check on imp (smart reader at least 0.9 of original): 6 of 6 gated papers.

## Judge ratings (mean over papers and samples)

| prompt | reader | orig | rem (minus orig; papers within 1) | imp (minus orig; papers within 1) |
|---|---|---|---|---|
| self | gpt-5.6-sol | 9.03 | 8.50 (-0.53; 10 of 12) | 9.10 (+0.07; 7 of 7) |
| self | gemma-4-31b-it | 10.00 | 10.00 (+0.00; 12 of 12) | 10.00 (+0.00; 7 of 7) |
| self | ministral-8b-2512 | 4.50 | 4.86 (+0.36; 10 of 12) | 4.14 (-0.36; 6 of 7) |
| self | llama-3.2-3b-instruct | 4.75 | 4.42 (-0.33; 9 of 12) | 4.57 (-0.18; 5 of 7) |
| human | gpt-5.6-sol | 7.69 | 6.00 (-1.69; 5 of 12) | 7.76 (+0.07; 7 of 7) |
| human | gemma-4-31b-it | 9.69 | 9.86 (+0.17; 12 of 12) | 9.48 (-0.22; 6 of 7) |
| human | ministral-8b-2512 | 7.50 | 8.06 (+0.56; 11 of 12) | 7.29 (-0.21; 6 of 7) |
| human | llama-3.2-3b-instruct | 6.14 | 6.53 (+0.39; 9 of 12) | 6.09 (-0.04; 5 of 7) |

## Per paper: quiz score orig / rem / imp

| paper | kept | gated | length rem, imp | gpt-5.6-sol | gemma-4-31b-it | ministral-8b-2512 | llama-3.2-3b-instruct |
|---|---|---|---|---|---|---|---|
| 2607.10112v1 | 11 | no | 1.036, 0.805 | 0.89 / 0.41 / 0.79 | 0.74 / 0.39 / 0.65 | 0.67 / 0.48 / 0.65 | 0.23 / 0.23 / 0.29 |
| 2607.17117v2 | 38 | yes | 1.095, - | 0.79 / 0.39 / - | 0.49 / 0.22 / - | 0.39 / 0.22 / - | 0.14 / 0.13 / - |
| 2607.20379v2 | 63 | yes | 1.025, - | 0.75 / 0.50 / - | 0.56 / 0.36 / - | 0.40 / 0.34 / - | 0.15 / 0.14 / - |
| 2607.28685v1 | 71 | yes | 0.914, 0.819 | 0.83 / 0.40 / 0.83 | 0.58 / 0.23 / 0.57 | 0.35 / 0.23 / 0.32 | 0.11 / 0.13 / 0.12 |
| 2608.13754v1 | 61 | yes | 0.893, - | 0.78 / 0.43 / - | 0.51 / 0.12 / - | 0.34 / 0.19 / - | 0.12 / 0.09 / - |
| 2608.19216v1 | 42 | yes | 1.043, 0.823 | 0.75 / 0.43 / 0.78 | 0.62 / 0.30 / 0.62 | 0.56 / 0.35 / 0.49 | 0.25 / 0.20 / 0.26 |
| 2608.25490v1 | 95 | yes | 0.857, - | 0.85 / 0.29 / - | 0.57 / 0.11 / - | 0.35 / 0.20 / - | 0.13 / 0.13 / - |
| 2608.30724v1 | 85 | yes | 1.042, 0.8 | 0.82 / 0.46 / 0.80 | 0.44 / 0.23 / 0.48 | 0.37 / 0.28 / 0.34 | 0.18 / 0.14 / 0.16 |
| 2609.00755v1 | 80 | yes | 1.021, - | 0.78 / 0.37 / - | 0.42 / 0.25 / - | 0.31 / 0.24 / - | 0.18 / 0.17 / - |
| 2609.02893v1 | 53 | yes | 1.064, 0.835 | 0.75 / 0.46 / 0.73 | 0.40 / 0.30 / 0.43 | 0.35 / 0.31 / 0.34 | 0.24 / 0.18 / 0.20 |
| 2609.30094v2 | 73 | yes | 1.113, 0.875 | 0.83 / 0.33 / 0.83 | 0.58 / 0.22 / 0.54 | 0.21 / 0.14 / 0.23 | 0.08 / 0.10 / 0.10 |
| 2609.34091v1 | 44 | yes | 1.08, 0.903 | 0.78 / 0.42 / 0.75 | 0.42 / 0.20 / 0.43 | 0.31 / 0.20 / 0.29 | 0.12 / 0.12 / 0.13 |
