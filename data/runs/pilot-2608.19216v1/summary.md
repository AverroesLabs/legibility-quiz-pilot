# Summary, grader z-ai/glm-5.3

## Mean score by reader and condition (mean over items and samples; share missing in brackets)

| Reader | none | paper | obf | rem | retrieved |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 0.00 (50%) | 0.83 (0%) | 0.83 (0%) | 0.83 (0%) |  |
| mistralai/ministral-8b-2512 | 0.20 (2%) | 0.82 (0%) | 0.76 (0%) | 0.71 (2%) |  |
| openai/gpt-5.6-sol | 0.49 (0%) | 0.91 (0%) | 0.90 (0%) | 0.92 (0%) | 0.34 (0%) |

## Per item (mean score, all readers in the row group, primary grader)

### Writer anthropic/claude-opus-5.5, seed 1

| Item | kind | form | gemma-4-31b-:non | gemma-4-31b-:pap | ministral-8b:non | ministral-8b:pap | gpt-5.6-sol:non | gpt-5.6-sol:pap |
|---|---|---|---|---|---|---|---|---|
| q1 | cla | apply-new- | 0.00 | 0.67 | 0.17 | 0.00 | 0.33 | 0.67 |
| q2 | cla | apply-new- | 0.00 | 1.00 | 0.17 | 1.00 | 1.00 | 1.00 |
| q3 | cla | spot-wrong | 0.00 | 0.33 | 0.67 | 0.67 | 0.50 | 0.67 |
| q4 | cla | change-con | 0.00 | 0.67 | 0.17 | 1.00 | 0.67 | 0.67 |
| q5 | cla | change-con | 0.00 | 0.33 | 0.00 | 0.33 | 0.00 | 1.00 |
| q6 | cla | connect-se | 0.00 | 1.00 | 0.25 | 1.00 | 0.50 | 1.00 |
| q7 | cla | connect-se | 0.00 | 1.00 | 0.25 | 1.00 | 1.00 | 1.00 |
| q8 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 0.83 | 0.00 | 1.00 |
| q9 | cla | connect-se | 0.00 | 0.50 | 0.00 | 0.75 | 0.25 | 0.50 |
| q10 | cla | connect-se | 0.00 | 1.00 | 0.25 | 0.75 | 0.50 | 0.50 |
| q11 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 0.50 | 0.50 | 1.00 |
| q12 | cla | spot-wrong | 0.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| q13 | cla | connect-se | 0.00 | 0.75 | 0.50 | 0.75 | 0.50 | 1.00 |
| q14 | cla | spot-wrong | 0.00 | 0.50 | 0.25 | 0.50 | 0.75 | 1.00 |
| q15 | cla | connect-se | 0.00 | 1.00 | 0.25 | 1.00 | 0.50 | 1.00 |
| q16 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q17 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.50 | 1.00 |
| q18 | cla | apply-new- | 0.00 | 0.83 | 0.50 | 0.67 | 0.67 | 1.00 |
| q19 | cla | connect-se | 0.00 | 0.83 | 0.33 | 1.00 | 0.67 | 1.00 |
| q20 | cla | connect-se | 0.00 | 0.33 | 0.00 | 0.50 | 0.00 | 0.67 |
| q21 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 0.75 | 0.75 | 1.00 |
| q22 | int | connect-se | 0.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| q23 | int | connect-se | 0.00 | 1.00 | 0.33 | 1.00 | 0.67 | 1.00 |
| q24 | int | connect-se | 0.00 | 1.00 | 0.00 | 1.00 | 0.33 | 1.00 |
| q25 | int | connect-se | 0.00 | 1.00 | 0.00 | 1.00 | 0.67 | 1.00 |

## Four-condition filter (development thresholds, not preregistered)

Smart takers: openai/gpt-5.6-sol. Kept 16 of 25 items.

| Item | smart none | smart paper | ladder none | outcome |
|---|---|---|---|---|
| q1 | 0.33 | 0.67 | 0.08 | kept |
| q2 | 1.00 | 1.00 | 0.08 | smart-none |
| q3 | 0.50 | 0.67 | 0.33 | kept |
| q4 | 0.67 | 0.67 | 0.08 | smart-none |
| q5 | 0.00 | 1.00 | 0.00 | kept |
| q6 | 0.50 | 1.00 | 0.12 | kept |
| q7 | 1.00 | 1.00 | 0.12 | smart-none |
| q8 | 0.00 | 1.00 | 0.00 | kept |
| q9 | 0.25 | 0.50 | 0.00 | kept |
| q10 | 0.50 | 0.50 | 0.12 | kept |
| q11 | 0.50 | 1.00 | 0.00 | kept |
| q12 | 0.50 | 1.00 | 0.25 | kept |
| q13 | 0.50 | 1.00 | 0.25 | kept |
| q14 | 0.75 | 1.00 | 0.12 | smart-none |
| q15 | 0.50 | 1.00 | 0.12 | kept |
| q16 | 0.00 | 1.00 | 0.00 | kept |
| q17 | 0.50 | 1.00 | 0.00 | kept |
| q18 | 0.67 | 1.00 | 0.25 | smart-none |
| q19 | 0.67 | 1.00 | 0.17 | smart-none |
| q20 | 0.00 | 0.67 | 0.00 | kept |
| q21 | 0.75 | 1.00 | 0.00 | smart-none |
| q22 | 0.50 | 1.00 | 0.17 | kept |
| q23 | 0.67 | 1.00 | 0.17 | smart-none |
| q24 | 0.33 | 1.00 | 0.00 | kept |
| q25 | 0.67 | 1.00 | 0.00 | smart-none |

Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):

| Reader | none | paper | gain |
|---|---|---|---|
| google/gemma-4-31b-it | 0.00 [0.00, 0.00] | 0.81 [0.67, 0.93] | 0.81 |
| mistralai/ministral-8b-2512 | 0.19 [0.09, 0.31] | 0.76 [0.61, 0.88] | 0.56 |
| openai/gpt-5.6-sol | 0.34 [0.23, 0.43] | 0.88 [0.78, 0.96] | 0.54 |
