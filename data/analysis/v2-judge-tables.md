Ratings 1 to 10, 12 core papers x 3 samples; 0 failed.

| Reader | Rating: orig | obf | comp | compobf | Judge: orig above obf / comp / compobf (papers) | Quiz: same | Judge, sample pairs | Quiz, run pairs | Spearman orig rating vs orig quiz (p) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | 9.03 | 7.50 | 9.00 | 7.06 | 1.00 / 0.54 / 1.00 | 0.75 / 0.58 / 0.83 | 1.00 / 0.51 / 1.00 | 0.58 / 0.59 / 0.76 | +0.22 (0.49) |
| gemma-4-31b-it | 10.00 | 7.94 | 10.00 | 6.14 | 1.00 / 0.50 / 1.00 | 0.88 / 0.58 / 0.92 | 0.85 / 0.50 / 1.00 | 0.82 / 0.62 / 0.88 | n/a (constant) |
| ministral-8b-2512 | 4.50 | 2.75 | 3.81 | 2.33 | 1.00 / 0.83 / 1.00 | 0.83 / 0.79 / 0.92 | 0.91 / 0.69 / 0.99 | 0.75 / 0.67 / 0.85 | -0.14 (0.67) |
| llama-3.2-3b-instruct | 4.75 | 4.14 | 4.44 | 4.00 | 0.75 / 0.62 / 0.83 | 0.67 / 0.88 / 0.75 | 0.69 / 0.59 / 0.72 | 0.62 / 0.71 / 0.68 | -0.10 (0.77) |

Spearman over papers, judge rating drop vs quiz score drop (orig minus version), rho (p):

| Reader | obf | comp | compobf |
| --- | --- | --- | --- |
| gpt-5.6-sol | -0.24 (0.45) | -0.13 (0.69) | +0.09 (0.79) |
| gemma-4-31b-it | -0.20 (0.54) | n/a (constant) | +0.23 (0.47) |
| ministral-8b-2512 | -0.27 (0.40) | -0.58 (0.05) | -0.54 (0.07) |
| llama-3.2-3b-instruct | +0.52 (0.09) | -0.06 (0.86) | -0.13 (0.69) |

Readability formulas: share of papers where the original reads easier than the version:

| Formula | obf | comp | compobf |
| --- | --- | --- | --- |
| fk_grade | 1.00 | 0.33 | 0.92 |
| flesch_ease | 0.92 | 0.58 | 0.83 |
| fog | 1.00 | 0.33 | 0.92 |
