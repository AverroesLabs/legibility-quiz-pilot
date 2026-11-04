Ratings 1 to 10, 12 core papers x 3 samples; 0 failed.

| Reader | Rating: orig | obf | comp | compobf | Judge: orig above obf / comp / compobf (papers) | Quiz: same | Judge, sample pairs | Quiz, run pairs | Spearman orig rating vs orig quiz (p) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | 7.69 | 4.28 | 7.42 | 3.94 | 1.00 / 0.54 / 1.00 | 0.75 / 0.58 / 0.83 | 1.00 / 0.60 / 1.00 | 0.58 / 0.59 / 0.76 | +0.11 (0.74) |
| gemma-4-31b-it | 9.69 | 2.92 | 8.89 | 2.11 | 1.00 / 0.67 / 1.00 | 0.88 / 0.58 / 0.92 | 1.00 / 0.62 / 1.00 | 0.82 / 0.62 / 0.88 | -0.02 (0.95) |
| ministral-8b-2512 | 7.50 | 2.50 | 6.33 | 2.25 | 1.00 / 0.71 / 1.00 | 0.83 / 0.79 / 0.92 | 1.00 / 0.65 / 1.00 | 0.75 / 0.67 / 0.85 | -0.08 (0.80) |
| llama-3.2-3b-instruct | 6.14 | 5.11 | 5.44 | 4.19 | 0.83 / 0.83 / 0.92 | 0.67 / 0.88 / 0.75 | 0.76 / 0.69 / 0.87 | 0.62 / 0.71 / 0.68 | +0.28 (0.39) |

Spearman over papers, judge rating drop vs quiz score drop (orig minus version), rho (p):

| Reader | obf | comp | compobf |
| --- | --- | --- | --- |
| gpt-5.6-sol | -0.45 (0.14) | -0.24 (0.46) | -0.05 (0.89) |
| gemma-4-31b-it | +0.28 (0.37) | +0.41 (0.19) | -0.02 (0.95) |
| ministral-8b-2512 | -0.16 (0.62) | +0.55 (0.06) | +0.03 (0.92) |
| llama-3.2-3b-instruct | +0.06 (0.86) | +0.15 (0.64) | +0.24 (0.46) |

Readability formulas: share of papers where the original reads easier than the version:

| Formula | obf | comp | compobf |
| --- | --- | --- | --- |
| fk_grade | 1.00 | 0.33 | 0.92 |
| flesch_ease | 0.92 | 0.58 | 0.83 |
| fog | 1.00 | 0.33 | 0.92 |
