# Summary, grader z-ai/glm-5.3

## Mean score by reader and condition (mean over items and samples; share missing in brackets)

| Reader | none | paper | obf | rem | retrieved |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 0.00 (0%) | 0.79 (0%) | 0.76 (0%) | 0.77 (0%) |  |
| mistralai/ministral-8b-2512 | 0.16 (0%) | 0.73 (0%) | 0.72 (0%) | 0.73 (0%) |  |
| openai/gpt-5.6-sol | 0.37 (0%) | 0.92 (0%) | 0.90 (0%) | 0.87 (0%) | 0.60 (0%) |

## Per item (mean score, all readers in the row group, primary grader)

### Writer anthropic/claude-opus-5.5, seed 1

| Item | kind | form | gemma-4-31b-:non | gemma-4-31b-:pap | ministral-8b:non | ministral-8b:pap | gpt-5.6-sol:non | gpt-5.6-sol:pap |
|---|---|---|---|---|---|---|---|---|
| q1 | cla | apply-new- | 0.00 | 0.67 | 0.00 | 0.50 | 0.33 | 0.67 |
| q2 | cla | spot-wrong | 0.00 | 0.33 | 0.00 | 0.67 | 0.33 | 0.83 |
| q3 | cla | connect-se | 0.00 | 1.00 | 0.00 | 1.00 | 0.50 | 1.00 |
| q4 | cla | spot-wrong | 0.00 | 0.67 | 0.33 | 0.83 | 0.67 | 0.67 |
| q5 | cla | change-con | 0.00 | 0.67 | 0.00 | 0.67 | 0.67 | 1.00 |
| q6 | cla | connect-se | 0.00 | 1.00 | 0.17 | 1.00 | 0.67 | 1.00 |
| q7 | cla | apply-new- | 0.00 | 1.00 | 0.33 | 0.50 | 0.33 | 1.00 |
| q8 | cla | spot-wrong | 0.00 | 0.67 | 0.00 | 0.67 | 0.00 | 0.83 |
| q9 | cla | apply-new- | 0.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| q10 | cla | change-con | 0.00 | 0.83 | 0.50 | 0.33 | 0.33 | 1.00 |
| q11 | cla | spot-wrong | 0.00 | 0.67 | 0.00 | 0.17 | 0.00 | 0.67 |
| q12 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 0.75 | 0.00 | 1.00 |
| q13 | cla | change-con | 0.00 | 1.00 | 0.25 | 1.00 | 0.50 | 1.00 |
| q14 | cla | spot-wrong | 0.00 | 1.00 | 0.00 | 0.67 | 0.33 | 0.67 |
| q15 | cla | apply-new- | 0.00 | 1.00 | 0.17 | 0.83 | 0.33 | 1.00 |
| q16 | cla | spot-wrong | 0.00 | 1.00 | 0.25 | 1.00 | 0.25 | 1.00 |
| q17 | cla | connect-se | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q18 | cla | spot-wrong | 0.00 | 1.00 | 0.17 | 1.00 | 0.50 | 1.00 |
| q19 | cla | spot-wrong | 0.00 | 0.50 | 0.00 | 1.00 | 0.50 | 1.00 |
| q20 | cla | change-con | 0.00 | 1.00 | 0.50 | 0.75 | 0.50 | 1.00 |
| q21 | cla | change-con | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.75 |
| q22 | int | apply-new- | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 |
| q23 | int | apply-new- | 0.00 | 0.83 | 0.00 | 0.67 | 0.33 | 1.00 |
| q24 | int | connect-se | 0.00 | 0.67 | 0.17 | 0.50 | 0.17 | 1.00 |
| q25 | int | apply-new- | 0.00 | 0.33 | 0.00 | 0.83 | 0.67 | 0.83 |

## Four-condition filter (development thresholds, not preregistered)

Smart takers: openai/gpt-5.6-sol. Kept 20 of 25 items.

| Item | smart none | smart paper | ladder none | outcome |
|---|---|---|---|---|
| q1 | 0.33 | 0.67 | 0.00 | kept |
| q2 | 0.33 | 0.83 | 0.00 | kept |
| q3 | 0.50 | 1.00 | 0.00 | kept |
| q4 | 0.67 | 0.67 | 0.17 | smart-none |
| q5 | 0.67 | 1.00 | 0.00 | smart-none |
| q6 | 0.67 | 1.00 | 0.08 | smart-none |
| q7 | 0.33 | 1.00 | 0.17 | kept |
| q8 | 0.00 | 0.83 | 0.00 | kept |
| q9 | 1.00 | 1.00 | 0.38 | smart-none |
| q10 | 0.33 | 1.00 | 0.25 | kept |
| q11 | 0.00 | 0.67 | 0.00 | kept |
| q12 | 0.00 | 1.00 | 0.00 | kept |
| q13 | 0.50 | 1.00 | 0.12 | kept |
| q14 | 0.33 | 0.67 | 0.00 | kept |
| q15 | 0.33 | 1.00 | 0.08 | kept |
| q16 | 0.25 | 1.00 | 0.12 | kept |
| q17 | 0.00 | 1.00 | 0.00 | kept |
| q18 | 0.50 | 1.00 | 0.08 | kept |
| q19 | 0.50 | 1.00 | 0.00 | kept |
| q20 | 0.50 | 1.00 | 0.25 | kept |
| q21 | 0.50 | 0.75 | 0.25 | kept |
| q22 | 0.00 | 1.00 | 0.00 | kept |
| q23 | 0.33 | 1.00 | 0.00 | kept |
| q24 | 0.17 | 1.00 | 0.08 | kept |
| q25 | 0.67 | 0.83 | 0.00 | smart-none |

Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):

| Reader | none | paper | gain |
|---|---|---|---|
| google/gemma-4-31b-it | 0.00 [0.00, 0.00] | 0.81 [0.71, 0.90] | 0.81 |
| mistralai/ministral-8b-2512 | 0.14 [0.07, 0.23] | 0.70 [0.59, 0.80] | 0.56 |
| openai/gpt-5.6-sol | 0.29 [0.20, 0.37] | 0.92 [0.86, 0.97] | 0.63 |
