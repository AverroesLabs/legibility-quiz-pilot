# Summary, grader z-ai/glm-5.3

## Mean score by reader and condition (mean over items and samples; share missing in brackets)

| Reader | none | paper | obf | rem | retrieved |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 0.00 (0%) | 0.74 (0%) | 0.76 (0%) | 0.81 (0%) |  |
| mistralai/ministral-8b-2512 | 0.24 (0%) | 0.63 (0%) | 0.58 (0%) | 0.62 (0%) |  |
| openai/gpt-5.6-sol | 0.50 (0%) | 0.96 (0%) | 0.96 (0%) | 0.97 (0%) | 0.48 (0%) |

## Per item (mean score, all readers in the row group, primary grader)

### Writer anthropic/claude-opus-5.5, seed 1

| Item | kind | form | gemma-4-31b-:non | gemma-4-31b-:pap | ministral-8b:non | ministral-8b:pap | gpt-5.6-sol:non | gpt-5.6-sol:pap |
|---|---|---|---|---|---|---|---|---|
| q1 | cla | apply-new- | 0.00 | 0.67 | 0.00 | 0.50 | 0.00 | 0.67 |
| q2 | cla | spot-wrong | 0.00 | 0.17 | 0.00 | 0.67 | 0.33 | 1.00 |
| q3 | cla | apply-new- | 0.00 | 1.00 | 0.33 | 0.83 | 0.67 | 1.00 |
| q4 | cla | connect-se | 0.00 | 0.83 | 0.17 | 0.50 | 0.50 | 1.00 |
| q5 | cla | apply-new- | 0.00 | 0.83 | 0.33 | 0.67 | 0.33 | 1.00 |
| q6 | cla | apply-new- | 0.00 | 0.75 | 0.00 | 1.00 | 0.25 | 1.00 |
| q7 | cla | apply-new- | 0.00 | 0.50 | 0.33 | 0.50 | 0.50 | 0.83 |
| q8 | cla | spot-wrong | 0.00 | 0.67 | 0.00 | 0.33 | 0.67 | 1.00 |
| q9 | cla | connect-se | 0.00 | 0.67 | 0.00 | 1.00 | 0.67 | 1.00 |
| q10 | cla | apply-new- | 0.00 | 0.67 | 0.17 | 0.67 | 0.50 | 1.00 |
| q11 | cla | connect-se | 0.00 | 0.50 | 0.33 | 0.67 | 0.67 | 1.00 |
| q12 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q13 | cla | spot-wrong | 0.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| q14 | cla | apply-new- | 0.00 | 1.00 | 0.33 | 0.67 | 1.00 | 1.00 |
| q15 | cla | apply-new- | 0.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 |
| q16 | cla | connect-se | 0.00 | 0.17 | 0.00 | 0.33 | 0.00 | 0.83 |
| q17 | cla | connect-se | 0.00 | 0.75 | 0.25 | 0.50 | 0.50 | 1.00 |
| q18 | cla | change-con | 0.00 | 0.50 | 0.25 | 0.00 | 0.75 | 1.00 |
| q19 | cla | change-con | 0.00 | 0.83 | 0.00 | 0.17 | 0.00 | 1.00 |
| q20 | cla | spot-wrong | 0.00 | 1.00 | 0.25 | 1.00 | 0.50 | 1.00 |
| q21 | cla | apply-new- | 0.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| q22 | int | connect-se | 0.00 | 1.00 | 0.33 | 0.83 | 0.67 | 1.00 |
| q23 | int | connect-se | 0.00 | 0.83 | 0.33 | 0.50 | 0.33 | 1.00 |
| q24 | int | connect-se | 0.00 | 0.00 | 0.00 | 0.17 | 0.67 | 0.67 |

## Four-condition filter (development thresholds, not preregistered)

Smart takers: openai/gpt-5.6-sol. Kept 14 of 24 items.

| Item | smart none | smart paper | ladder none | outcome |
|---|---|---|---|---|
| q1 | 0.00 | 0.67 | 0.00 | kept |
| q2 | 0.33 | 1.00 | 0.00 | kept |
| q3 | 0.67 | 1.00 | 0.17 | smart-none |
| q4 | 0.50 | 1.00 | 0.08 | kept |
| q5 | 0.33 | 1.00 | 0.17 | kept |
| q6 | 0.25 | 1.00 | 0.00 | kept |
| q7 | 0.50 | 0.83 | 0.17 | kept |
| q8 | 0.67 | 1.00 | 0.00 | smart-none |
| q9 | 0.67 | 1.00 | 0.00 | smart-none |
| q10 | 0.50 | 1.00 | 0.08 | kept |
| q11 | 0.67 | 1.00 | 0.17 | smart-none |
| q12 | 0.00 | 1.00 | 0.00 | kept |
| q13 | 1.00 | 1.00 | 0.38 | smart-none |
| q14 | 1.00 | 1.00 | 0.17 | smart-none |
| q15 | 1.00 | 1.00 | 0.50 | smart-none |
| q16 | 0.00 | 0.83 | 0.00 | kept |
| q17 | 0.50 | 1.00 | 0.12 | kept |
| q18 | 0.75 | 1.00 | 0.12 | smart-none |
| q19 | 0.00 | 1.00 | 0.00 | kept |
| q20 | 0.50 | 1.00 | 0.12 | kept |
| q21 | 0.50 | 1.00 | 0.25 | kept |
| q22 | 0.67 | 1.00 | 0.17 | smart-none |
| q23 | 0.33 | 1.00 | 0.17 | kept |
| q24 | 0.67 | 0.67 | 0.00 | smart-none |

Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):

| Reader | none | paper | gain |
|---|---|---|---|
| google/gemma-4-31b-it | 0.00 [0.00, 0.00] | 0.71 [0.57, 0.85] | 0.71 |
| mistralai/ministral-8b-2512 | 0.17 [0.08, 0.26] | 0.64 [0.51, 0.77] | 0.48 |
| openai/gpt-5.6-sol | 0.30 [0.19, 0.40] | 0.95 [0.89, 1.00] | 0.65 |
