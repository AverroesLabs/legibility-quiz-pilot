# Summary, grader z-ai/glm-5.3

## Mean score by reader and condition (mean over items and samples; share missing in brackets)

| Reader | none | paper | obf | rem | retrieved |
|---|---|---|---|---|---|
| google/gemma-4-31b-it | 0.00 (100%) | 0.83 (2%) | 0.86 (0%) | 0.58 (0%) |  |
| mistralai/ministral-8b-2512 | 0.15 (0%) | 0.65 (0%) | 0.57 (0%) | 0.44 (0%) |  |
| openai/gpt-5.6-sol | 0.27 (0%) | 0.96 (0%) | 0.97 (0%) | 0.65 (0%) | 0.26 (0%) |

## Per item (mean score, all readers in the row group, primary grader)

### Writer anthropic/claude-opus-5.5, seed 1

| Item | kind | form | gemma-4-31b-:non | gemma-4-31b-:pap | ministral-8b:non | ministral-8b:pap | gpt-5.6-sol:non | gpt-5.6-sol:pap |
|---|---|---|---|---|---|---|---|---|
| q1 | cla | apply-new- | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 |
| q2 | cla | connect-se | 0.00 | 1.00 | 0.00 | 1.00 | 0.50 | 1.00 |
| q3 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q4 | cla | apply-new- | 0.00 | 1.00 | 0.33 | 0.83 | 0.33 | 1.00 |
| q5 | cla | apply-new- | 0.00 | 0.67 | 0.00 | 0.50 | 0.50 | 0.83 |
| q6 | cla | spot-wrong | 0.00 | 0.67 | 0.00 | 0.17 | 0.17 | 0.67 |
| q7 | cla | spot-wrong | 0.00 | 0.67 | 0.33 | 0.50 | 0.17 | 1.00 |
| q8 | cla | apply-new- | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 1.00 |
| q9 | cla | apply-new- | 0.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| q10 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 0.50 | 0.25 | 1.00 |
| q11 | cla | connect-se | 0.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 |
| q12 | cla | spot-wrong | 0.00 | 1.00 | 0.00 | 0.75 | 0.00 | 1.00 |
| q13 | cla | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.25 | 1.00 |
| q14 | cla | spot-wrong | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| q15 | cla | connect-se | 0.00 | 1.00 | 0.17 | 1.00 | 0.17 | 1.00 |
| q16 | cla | apply-new- | 0.00 | 0.25 | 0.00 | 0.75 | 0.00 | 1.00 |
| q17 | cla | spot-wrong | 0.00 | 0.50 | 0.33 | 0.33 | 0.00 | 1.00 |
| q18 | cla | connect-se | 0.00 | 1.00 | 0.50 | 0.25 | 0.50 | 1.00 |
| q19 | cla | connect-se | 0.00 | 1.00 | 0.25 | 1.00 | 0.50 | 1.00 |
| q20 | cla | change-con | 0.00 | 0.83 | 0.00 | 0.50 | 0.33 | 1.00 |
| q21 | cla | connect-se | 0.00 | 1.00 | 0.33 | 0.50 | 0.67 | 1.00 |
| q22 | cla | apply-new- | 0.00 | 0.50 | 0.00 | 1.00 | 0.25 | 1.00 |
| q23 | cla | connect-se | 0.00 | 0.50 | 0.25 | 0.25 | 0.00 | 1.00 |
| q24 | cla | change-con | 0.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| q25 | int | connect-se | 0.00 | 0.75 | 0.00 | 0.50 | 0.50 | 1.00 |
| q26 | int | spot-wrong | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| q27 | int | apply-new- | 0.00 | 1.00 | 0.00 | 1.00 | 0.25 | 1.00 |
| q28 | int | connect-se | 0.00 | 0.50 | 0.00 | 0.25 | 0.25 | 0.50 |

## Four-condition filter (development thresholds, not preregistered)

Smart takers: openai/gpt-5.6-sol. Kept 26 of 28 items.

| Item | smart none | smart paper | ladder none | outcome |
|---|---|---|---|---|
| q1 | 0.00 | 1.00 | 0.00 | kept |
| q2 | 0.50 | 1.00 | 0.00 | kept |
| q3 | 0.00 | 1.00 | 0.00 | kept |
| q4 | 0.33 | 1.00 | 0.17 | kept |
| q5 | 0.50 | 0.83 | 0.00 | kept |
| q6 | 0.17 | 0.67 | 0.00 | kept |
| q7 | 0.17 | 1.00 | 0.17 | kept |
| q8 | 0.00 | 1.00 | 0.25 | kept |
| q9 | 0.50 | 1.00 | 0.25 | kept |
| q10 | 0.25 | 1.00 | 0.00 | kept |
| q11 | 0.50 | 1.00 | 0.25 | kept |
| q12 | 0.00 | 1.00 | 0.00 | kept |
| q13 | 0.25 | 1.00 | 0.00 | kept |
| q14 | 0.00 | 1.00 | 0.00 | kept |
| q15 | 0.17 | 1.00 | 0.08 | kept |
| q16 | 0.00 | 1.00 | 0.00 | kept |
| q17 | 0.00 | 1.00 | 0.17 | kept |
| q18 | 0.50 | 1.00 | 0.25 | kept |
| q19 | 0.50 | 1.00 | 0.12 | kept |
| q20 | 0.33 | 1.00 | 0.00 | kept |
| q21 | 0.67 | 1.00 | 0.17 | smart-none |
| q22 | 0.25 | 1.00 | 0.00 | kept |
| q23 | 0.00 | 1.00 | 0.12 | kept |
| q24 | 1.00 | 1.00 | 0.12 | smart-none |
| q25 | 0.50 | 1.00 | 0.00 | kept |
| q26 | 0.00 | 1.00 | 0.00 | kept |
| q27 | 0.25 | 1.00 | 0.00 | kept |
| q28 | 0.25 | 0.50 | 0.00 | kept |

Score on kept items (mean over items of the per-item mean; 95% bootstrap CI over items):

| Reader | none | paper | gain |
|---|---|---|---|
| google/gemma-4-31b-it | 0.00 [0.00, 0.00] | 0.82 [0.73, 0.91] | 0.82 |
| mistralai/ministral-8b-2512 | 0.14 [0.07, 0.22] | 0.66 [0.54, 0.78] | 0.52 |
| openai/gpt-5.6-sol | 0.23 [0.15, 0.30] | 0.96 [0.91, 1.00] | 0.73 |
