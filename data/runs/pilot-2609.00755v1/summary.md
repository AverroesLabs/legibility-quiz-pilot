# Summary, grader z-ai/glm-5.3

## Mean score by reader and condition (mean over items and samples; share missing in brackets)

| Reader | none | paper | obf | rem | retrieved |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 0.16 (0%) | 0.80 (0%) | 0.71 (0%) | 0.68 (0%) |  |
| mistralai/ministral-8b-2512 | 0.22 (0%) | 0.64 (0%) | 0.68 (0%) | 0.62 (0%) |  |
| openai/gpt-5.6-sol | 0.62 (0%) | 0.98 (0%) | 0.99 (0%) | 0.94 (0%) | 0.45 (0%) |

## Per item (mean score, all readers in the row group, primary grader)

### Writer anthropic/claude-opus-5.5, seed 1

| Item | kind | form | gemma-4-31b-:non | gemma-4-31b-:pap | ministral-8b:non | ministral-8b:pap | gpt-5.6-sol:non | gpt-5.6-sol:pap |
|---|---|---|---|---|---|---|---|---|
| q1 | cla | change-con | 0.50 | 1.00 | 0.50 | 1.00 | 0.75 | 1.00 |
| q2 | cla | connect-se | 0.33 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| q3 | cla | apply-new- | 0.17 | 0.67 | 0.17 | 0.50 | 0.33 | 0.67 |
| q4 | cla | apply-new- | 0.17 | 1.00 | 0.17 | 1.00 | 1.00 | 1.00 |
| q5 | cla | spot-wrong | 0.17 | 1.00 | 0.50 | 0.83 | 1.00 | 1.00 |
| q6 | cla | spot-wrong | 0.00 | 0.75 | 0.50 | 0.50 | 1.00 | 0.75 |
| q7 | cla | apply-new- | 0.00 | 0.50 | 0.17 | 0.00 | 1.00 | 1.00 |
| q8 | cla | change-con | 0.00 | 1.00 | 0.00 | 1.00 | 0.25 | 1.00 |
| q9 | cla | connect-se | 0.25 | 1.00 | 0.25 | 1.00 | 1.00 | 1.00 |
| q10 | cla | apply-new- | 0.00 | 1.00 | 0.17 | 0.17 | 0.33 | 1.00 |
| q11 | cla | connect-se | 0.33 | 0.83 | 0.00 | 0.00 | 0.67 | 1.00 |
| q12 | cla | spot-wrong | 0.25 | 0.50 | 0.25 | 0.25 | 0.75 | 1.00 |
| q13 | cla | apply-new- | 0.25 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| q14 | cla | change-con | 0.25 | 0.75 | 0.00 | 0.50 | 0.50 | 1.00 |
| q15 | cla | apply-new- | 0.25 | 0.75 | 0.00 | 1.00 | 0.50 | 1.00 |
| q16 | cla | apply-new- | 0.25 | 1.00 | 0.00 | 1.00 | 0.25 | 1.00 |
| q17 | cla | connect-se | 0.17 | 0.83 | 0.67 | 0.83 | 0.83 | 1.00 |
| q18 | cla | connect-se | 0.25 | 1.00 | 0.00 | 0.75 | 0.50 | 1.00 |
| q19 | cla | spot-wrong | 0.25 | 0.50 | 0.50 | 0.75 | 1.00 | 1.00 |
| q20 | cla | change-con | 0.00 | 0.75 | 0.00 | 0.50 | 0.25 | 1.00 |
| q21 | cla | connect-se | 0.25 | 0.50 | 0.00 | 0.25 | 0.50 | 1.00 |
| q22 | cla | change-con | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 |
| q23 | cla | apply-new- | 0.00 | 0.67 | 0.50 | 0.67 | 0.67 | 1.00 |
| q24 | cla | connect-se | 0.17 | 0.83 | 0.00 | 0.67 | 0.33 | 1.00 |
| q25 | cla | apply-new- | 0.25 | 1.00 | 0.25 | 0.75 | 0.25 | 1.00 |
| q26 | int | connect-se | 0.25 | 0.75 | 0.50 | 0.50 | 0.50 | 1.00 |
| q27 | int | apply-new- | 0.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| q28 | int | connect-se | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 |
| q29 | int | connect-se | 0.00 | 0.67 | 0.33 | 1.00 | 1.00 | 1.00 |

## Four-condition filter (development thresholds, not preregistered)

Smart takers: openai/gpt-5.6-sol. Kept 15 of 29 items.

| Item | smart none | smart paper | ladder none | outcome |
|---|---|---|---|---|
| q1 | 0.75 | 1.00 | 0.50 | smart-none |
| q2 | 1.00 | 1.00 | 0.33 | smart-none |
| q3 | 0.33 | 0.67 | 0.17 | kept |
| q4 | 1.00 | 1.00 | 0.17 | smart-none |
| q5 | 1.00 | 1.00 | 0.33 | smart-none |
| q6 | 1.00 | 0.75 | 0.25 | smart-none |
| q7 | 1.00 | 1.00 | 0.08 | smart-none |
| q8 | 0.25 | 1.00 | 0.00 | kept |
| q9 | 1.00 | 1.00 | 0.25 | smart-none |
| q10 | 0.33 | 1.00 | 0.08 | kept |
| q11 | 0.67 | 1.00 | 0.17 | smart-none |
| q12 | 0.75 | 1.00 | 0.25 | smart-none |
| q13 | 0.50 | 1.00 | 0.25 | kept |
| q14 | 0.50 | 1.00 | 0.12 | kept |
| q15 | 0.50 | 1.00 | 0.12 | kept |
| q16 | 0.25 | 1.00 | 0.12 | kept |
| q17 | 0.83 | 1.00 | 0.42 | smart-none |
| q18 | 0.50 | 1.00 | 0.12 | kept |
| q19 | 1.00 | 1.00 | 0.38 | smart-none |
| q20 | 0.25 | 1.00 | 0.00 | kept |
| q21 | 0.50 | 1.00 | 0.12 | kept |
| q22 | 0.00 | 1.00 | 0.00 | kept |
| q23 | 0.67 | 1.00 | 0.25 | smart-none |
| q24 | 0.33 | 1.00 | 0.08 | kept |
| q25 | 0.25 | 1.00 | 0.25 | kept |
| q26 | 0.50 | 1.00 | 0.38 | kept |
| q27 | 1.00 | 1.00 | 0.25 | smart-none |
| q28 | 0.00 | 1.00 | 0.00 | kept |
| q29 | 1.00 | 1.00 | 0.17 | smart-none |

Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):

| Reader | none | paper | gain |
|---|---|---|---|
| google/gemma-4-31b-it | 0.16 [0.09, 0.21] | 0.80 [0.70, 0.89] | 0.64 |
| mistralai/ministral-8b-2512 | 0.09 [0.02, 0.17] | 0.57 [0.43, 0.72] | 0.48 |
| openai/gpt-5.6-sol | 0.33 [0.24, 0.41] | 0.98 [0.93, 1.00] | 0.64 |
