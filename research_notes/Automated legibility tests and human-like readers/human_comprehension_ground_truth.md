# Human comprehension ground truth for validating an automated legibility test

Scope note: about 24 search, fetch and extraction calls. Several primary full texts were paywalled or blocked by captchas: the OneStop Scientific Data paper on PMC, the CLEAR paper on Springer, and McNamara 1996 / Britton and Gulgoz 1991. Where only an abstract or a secondary summary was available, the finding says so. Numbers marked "(abstract)" or "(secondary)" were not checked against full text.

## 1. Datasets with human comprehension scores on texts of varying difficulty

### Takeaway
No public dataset gives human comprehension scores for research papers written at different levels of clarity. The closest controlled resource is OneStopQA / OneStop eye-tracking: the same questions are asked about original and simplified versions of each news paragraph, with 360 human readers. QuALITY gives per-question human accuracy under a 45-second time limit versus untimed reading of long texts. CLEAR gives teacher-judged difficulty, not measured comprehension. For research papers specifically, the usable human data are small experiments (Freeling et al. 2021 abstracts; Kerwer et al. 2021 PLS), and their materials may need to be requested from the authors.

### Cited Findings
**OneStopQA / OneStopEnglish (STARC)**
- STARC/OneStopQA (Berzak, Malmaud and Levy, ACL 2020) uses Guardian articles from OneStopEnglish, each at Elementary, Intermediate and Advanced level. Each paragraph has 3 multiple-choice questions, and the same questions are answerable from all three levels. That makes it a ready-made design for testing whether "same content, different difficulty" changes accuracy. — [STARC arXiv](https://arxiv.org/abs/2004.14797v1); [GitHub](https://github.com/berzak/onestop-qa); [HF card](https://huggingface.co/datasets/onestop_qa)
- The human ceiling on OneStopQA is 95.3%, versus 84.7% on RACE, and was confirmed in an in-lab experiment. The paper also reports that in RACE-style datasets 47% of questions can be guessed by models without the passage and 18% are unanimously judged by humans to have no unique answer. — [STARC arXiv](https://arxiv.org/abs/2004.14797v1)
- The OneStop eye-tracking dataset (Scientific Data 2025) has 360 English L1 participants, 152 hours of EyeLink 1000 Plus recordings and 2.6M word tokens. Each participant reads 10 Guardian articles paragraph by paragraph and answers one comprehension question per paragraph. Paragraphs come in an Advanced (original) and an Elementary version simplified by experienced language teachers. There are 486 questions (3 per paragraph). Data are on OSF. — [OneStop site](https://lacclab.github.io/OneStop-Eye-Movements); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12727756/); [OSF](https://osf.io/2prdq)
- Pooled accuracy was about 81.2% (7,890/9,718 trials) in ordinary reading ("gathering") and 86.9% (8,450/9,720) when readers saw the question first ("hunting"). I could not get accuracy split by Advanced vs Elementary because PMC returned a captcha. — [arXiv 2410.04484 / related OneStop analyses, via search snippet](https://arxiv.org/pdf/2410.04484)
- A related paper, "Eye Tracking Based Cognitive Evaluation of Automatic Readability Assessment Methods" (arXiv 2502.11150), uses OneStop to judge readability formulas against human reading behaviour. That is methodologically the closest existing work to validating an automated legibility metric against human data. I did not open it. — [arXiv 2502.11150](https://arxiv.org/pdf/2502.11150)

**QuALITY**
- 6,737 multiple-choice questions on passages of 2k–8k tokens (mean about 5k). Sources are Project Gutenberg fiction (mostly sci-fi), Slate articles from the Open American National Corpus, and nonfiction from The Long+Short, Freesouls and the book *Open Access*, all CC-BY or more permissive. — [QuALITY, NAACL 2022](https://aclanthology.org/2022.naacl-main.391.pdf)
- Human accuracy (majority vote of untimed annotators) is 93.5% overall, 89.1% on HARD and 97.0% on EASY. The best baseline model scored 55.4% overall and 46.7% on HARD. — [QuALITY](https://aclanthology.org/2022.naacl-main.391.pdf)
- In speed validation, annotators see the text for only 45 seconds per question. HARD (3,360 questions) means the majority of timed annotators got the question wrong. 45 annotators were recruited. Speed-validation tasks paid $2.25 plus $0.20 per correct answer and took 11–12 minutes (about $17/hr). Untimed validation paid $6.50 plus bonuses, took 50–60 minutes, and came to about $15.96/hr. Writers earned about $21.05/hr. Total cost was about $9.10 per question. — [QuALITY](https://aclanthology.org/2022.naacl-main.391.pdf)
- Annotators rated HARD questions as answerable and unambiguous slightly less often than EASY ones (92.8% vs 95.1%). — [QuALITY](https://aclanthology.org/2022.naacl-main.391.pdf)

**CLEAR corpus (CommonLit)**
- About 5,000 excerpts for grades 3–12, spanning more than 250 years and two genres. The main label, "BT Easiness", is a Bradley-Terry coefficient (with standard error) fitted to teachers' pairwise difficulty judgments. The corpus also includes 9 traditional readability indices (ARTE) and predictions from 6 Kaggle prize winners. The dataset is MIT-licensed; passage licenses vary (mostly public domain, some CC BY / BY-SA / NC). Version 6.01 dates from December 2021. Student comprehension data are not mentioned. — [CommonLit blog](https://www.commonlit.org/blog/introducing-the-clear-corpus-an-open-dataset-to-advance-research-28ff8cfea84a/); [Crossley et al., BRM](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10027808/)

**PoTeC (German, scientific texts, experts vs novices)**
- 75 participants read 12 German scientific (textbook) texts in a 2×2×2 design: level of study (beginner/advanced) and discipline (physics/biology) between subjects, text domain within subjects. Each text has comprehension questions and text-independent background-knowledge questions. Eye tracking, data and code are on GitHub (published in BRM 2025). It is the only public dataset I found with comprehension scores for scientific text crossed with reader expertise. — [arXiv 2403.00506](https://arxiv.org/abs/2403.00506); [GitHub](https://github.com/DiLi-Lab/PoTeC)

**Lay-summary / plain-language studies with comprehension tests**
- Buljan et al. 2018 (J Clin Epidemiol 97:86–94) ran 3 parallel RCTs comparing an infographic with a text Cochrane plain-language summary, in students (n=171), consumers (n=99) and doctors (n=64). Knowledge did not differ in any trial; the infographic was preferred and rated more enjoyable. Trials were registered at NCT02918656 and NCT02980107, with protocols public. — [Liverpool repository](https://livrepository.liverpool.ac.uk/id/eprint/3025011); [CROSBI](https://www.bib.irb.hr/1053889); [ClinicalTrials.gov](https://clinicaltrials.gov/study/NCT02918656)
- Kerwer et al. 2021 (Frontiers in Psychology, "How to Put It Plainly?") ran two preregistered RCTs of plain-language summaries of psychology meta-analyses, with N=2,288 and N=2,211 stratified by education, age and gender. Outcomes were perceived accessibility, understanding, empowerment and knowledge acquisition. Explaining or replacing technical terms mattered, and excessive statistical detail hurt user experience. Materials and preregistrations are on PsychArchives. — [Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.771399/pdf); [PsychArchives](https://psycharchives.org/handle/20.500.12034/4230)
- Stricker et al. 2020 (PLoS ONE) compared PLS and scientific abstracts for 103 psychology articles using readability formulas only, with no human comprehension measure. — [DOAJ](https://doaj.org/article/fca64d2c761f4e36af5aa31fedde5135)

### Inferences
- OneStop is the best off-the-shelf check of whether an automated test ranks the Advanced version of a paragraph harder than the Elementary one, because it uses identical questions and humans answered them. But it is short news text written for ESL learners, not research papers.
- QuALITY's timed/untimed split works as a per-question "hard for skimming humans" label and is a reasonable target for a weaker-reader LLM quiz on long documents. It labels question difficulty, not prose quality.
- CLEAR labels are expert judgments of difficulty, not comprehension, so they validate perceived readability only.
- For research papers, ground truth will probably need a new human study. The Freeling 2021 and Kerwer 2021 designs are templates.

### Gaps
- OneStop accuracy by level (Advanced vs Elementary). The primary paper was blocked by a captcha; the per-level split should be in the OSF data.
- Newsela: not investigated. As far as I know it has no human comprehension scores and requires a license request, but this is unverified.
- eLife digests: I found no study with comprehension tests on them.
- I found no public dataset of human quiz accuracy on full research papers.

## 2. Effect sizes: coherence, cohesion, jargon, sentence complexity and organization

### Takeaway
Writing-quality effects on measured comprehension are real but usually small to moderate, and they depend heavily on reader knowledge and question type. They are much larger on subjective ratings (perceived readability, confidence) than on objective accuracy. Big effects show up mainly when the original text is genuinely deficient (Britton and Gulgoz principled revision) or in heavily reworked consent forms (pooled SMD about 1.7, secondary source). Cohesion manipulations of science text often move text-based accuracy by only a few percentage points, and can hurt high-knowledge readers.

### Cited Findings
- **Britton and Gulgoz 1991** (J Educ Psych 83:329–345, N=170 undergraduates). They used Kintsch's model to find the places in a 1,000-word expository text where readers must infer a missing link, and repaired each one. Free recall of this "principled revision" was "much increased" over the original. Readers' knowledge structures were also closer to the author's and experts' structures. Exact effect sizes were not available to me (abstract only). — [bibbase record](https://bibbase.org/network/publication/britton-glgz-usingkintschscomputationalmodeltoimproveinstructionaltexteffectsofrepairinginferencecallsonrecallandcognitivestructures-1991); [Document Design 1999 follow-up](https://benjamins.com/catalog/dd.1.2.04gul)
- A 2000 comparison of the Britton–Gulgoz procedure with a narrative-theory procedure found that only the second procedure improved inferential learning, and both had limited effects on recall. — [Document Design / Benjamins](https://benjamins.com/catalog/dd.1.2.04gul) (abstract-level, via search)
- **McNamara, Kintsch, Songer and Kintsch 1996** (Cognition and Instruction 14) and **McNamara and Kintsch 1996** (Discourse Processes). Texts were revised by increasing argument overlap, adding connectives and adding background information. Low-knowledge readers benefit from high cohesion. High-knowledge readers do worse with high-cohesion text on deep measures (inference questions, card sorting): the "reverse cohesion effect", explained as passive processing. One review says McNamara and Kintsch 1996 found high-knowledge readers better with low coherence and no effect for low-knowledge readers. — [Ozuru et al. 2005 CogSci, intro](https://escholarship.org/uc/item/0tw744hb); [search summary of reviews](https://sites.uw.edu/ccdl/files//2013/12/Long_Assessing-text_2006.pdf)
- O'Reilly and McNamara 2007 qualified this: high-cohesion text can help strategic high-knowledge readers. A recent preprint found no overall cohesion effects on comprehension. — [search summary](https://10wvvw.easychair.org/publications/preprint/4Krz/open)
- **Ozuru, Dempsey, Sayroo and McNamara 2005** (CogSci, N=108 low-knowledge undergraduates, within-subject high/low cohesion biology texts):
  - Cohesion had no effect on overall comprehension.
  - For text-based questions the benefit was marginal (t(107)=1.84, p=.06). Lower-knowledge participants gained significantly on text-based questions (t(52)=2.16, p=.035).
  - Cohesion had no effect on local or global inference questions.
  - Proportion correct on text-based questions moved from .20 to .22 in one reading-skill group and from .37 to .45 in the other.
  - The authors say the benefit was "quite small". The revision made the text about 50% longer. — [Ozuru et al. 2005](https://escholarship.org/uc/item/0tw744hb)
- Linderholm et al. 2000: causal-cohesion revision helped only on the more difficult text, regardless of reader skill. Voss and Silfies 1996: skilled readers gained more from expanded causal history texts. — [cited in Ozuru et al. 2005](https://escholarship.org/uc/item/0tw744hb)
- **Jargon (Bullock, Colón Amill, Shulman and Dixon 2019, Public Understanding of Science)**: N=650 experiment on emerging technologies. Jargon lowered processing fluency even when definitions were provided, and lower fluency increased resistance to persuasion and risk perception and decreased support. Outcomes were metacognitive (fluency and attitudes), not comprehension accuracy. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31354058/); [PDF](https://comm.osu.edu/sites/comm.osu.edu/files/PUS%202019-%20Bullock%20et%20al..pdf)
- Shulman and Bullock 2020 (PLOS ONE, "Don't dumb it down"): jargon increased processing difficulty and reduced persuasion for two less urgent topics, but had no effect in a COVID-19 condition. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540871)
- **Informed consent**:
  - Flory and Emanuel 2004 (JAMA): only 6 of 15 enhanced-consent-form trials improved understanding, and 5 of those 6 were of limited quality. — [Weill Cornell](https://vivo.weill.cornell.edu/display/pubid15467062)
  - Nishimura et al. 2013 (BMC Med Ethics): 39 studies and 54 interventions, with 22 meta-analysed. Pooled effects were SMD 1.73 for enhanced consent forms (significant; CI truncated in the snippet) and SMD 0.30 (95% CI −0.23 to 0.84) for multimedia, as reported in a secondary 2021 table. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/23879694/); [secondary table](https://0-bmcmedethics-biomedcentral-com.brum.beds.ac.uk/articles/10.1186/s12910-021-00639-x/tables/1)
  - A 2026 pediatric RCT found simplified and standard forms gave equivalent comprehension on a validated questionnaire. It notes that earlier positive comparisons often used non-validated questionnaires. — [Bioethics 2026](https://www.erudit.org/en/journals/bioethics/2026-v9-n2-bioethics010674/1124207ar/)
  - A 2025 systematic review covered 13,940 clinical consent forms and found about three-quarters hard to read. — [DOAJ](https://doaj.org/article/1fb26c9e5d0c4496ba485b32bb440425)
- I found no meta-analysis pooling cohesion or revision effects on expository comprehension. — [search outcome; see Ozuru 2005 review of mixed results](https://escholarship.org/uc/item/0tw744hb)

### Inferences
- If human effects of prose quality on multiple-choice accuracy are often 0–8 percentage points, the near-null response of LLM readers may be partly realistic. Validation needs adequate power (hundreds of reader-text observations) and may want subjective measures (perceived understanding, confidence, time) next to accuracy.
- Effects are concentrated in low-knowledge readers and in text-based or recall questions. A quiz aimed at deep inference may not register poor cohesion. Reverse cohesion effects mean "more explicit" is not monotonically better for expert readers.
- Reading time is a likely more sensitive human signal than accuracy, which supports eye-tracking or time datasets such as OneStop and PoTeC. This is an inference, not a measured comparison.

### Gaps
- Exact effect sizes for Britton and Gulgoz 1991 and McNamara et al. 1996 (paywalled).
- A quantitative meta-analysis of cohesion, jargon or sentence-complexity effects on comprehension accuracy was not found.
- The full Nishimura 2013 CI for enhanced forms.

## 3. Comprehension of real research papers, and clearly vs poorly written versions

### Takeaway
Only one controlled study came up: Freeling et al. 2021, with real science abstracts rewritten at 4 difficulty levels. Accessible writing raised perceived readability and confidence substantially, but objective understanding moved only modestly and non-monotonically. I found no human comprehension study on full research papers in clear versus poor versions.

### Cited Findings
- **Freeling, Doubleday, Dry, Semmler and Connell 2021** (Frontiers in Psychology 12:714321):
  - Materials: 3 real abstracts (social anxiety, solar cells, populist politics), each rewritten at 4 levels from Easiest to Hardest, with content and logic held constant. Nine features were varied, including narrator, punctuation, conjunctions, signposts, noun chunks, acronyms (0–15), hedges and word count (110–230).
  - Sample: 170 third-year science undergraduates gave 628 understanding responses and 347 readability/confidence responses.
  - Measures: 5 minutes per abstract, 2 multiple-choice questions each, about 30-minute sessions.
  - Understanding (% correct): Easiest 57.9, Easy 39.5, Hard 40.8, Hardest 48.0.
  - Readability ratings: 66.3 → 44.4 from Easiest to Hardest. Confidence: 62.4 → 48.8.
  - Statistics: readability slope 2.91 (95% CI 1.79–4.03); confidence slope 1.87; understanding OR 1.08 per level (CI 1.01–1.15, p=.027). Marginal R² for understanding was only 0.053, versus 0.244 for readability. Topic had large effects (populism OR 1.81 on understanding). The paper's text and Table 2 report slightly inconsistent confidence numbers. — [Frontiers](https://www.frontiersin.org/articles/10.3389/fpsyg.2021.714321/full)
- Plavén-Sigray et al. (eLife 2017, cited in the same search), a bibliometric study of more than 700,000 abstracts from 1881 to 2015: readability has declined, and more than a fifth of abstracts are at a level hard even for college graduates. This is formula-based with no comprehension test. — [Freeling et al. background, via search](https://pmc.ncbi.nlm.nih.gov/articles/PMC8430246)

### Inferences
- The Freeling design can be adapted to papers: same content, several writing levels, multiple-choice understanding, plus confidence and readability ratings. Their 2 questions per abstract is far too few for reliable item-level measurement. A legibility test would need many more questions per text.
- Freeling's data suggest an automated test might track perceived readability more easily than accuracy. Humans themselves show weak accuracy effects on short abstracts.

### Gaps
- No studies were found measuring researchers' or graduate students' quiz accuracy on full papers, or comparing well and poorly written full papers.
- Possible leads not checked: studies of reviewer comprehension, and "statistical reporting comprehension" studies.

## 4. Expert vs novice reading: background knowledge

### Takeaway
Background knowledge is consistently a larger driver of comprehension than text cohesion, and it interacts with it. In Ozuru 2005 the knowledge main effect was large while cohesion was non-significant overall. High-knowledge readers can be unhelped or even hurt by added cohesion. PoTeC is the main public dataset crossing expertise with scientific texts.

### Cited Findings
- Ozuru et al. 2005: main effect of domain knowledge F(1,106)=31.66, p<.001, and reading skill F(1,106)=23.47, versus no overall cohesion effect. — [Ozuru et al. 2005](https://escholarship.org/uc/item/0tw744hb)
- McNamara line of work: low-knowledge readers benefit from cohesion, while high-knowledge readers are hurt on deep measures. McNamara 2001 found the detrimental effect for high-knowledge readers only on text-based questions. — [Ozuru et al. 2005](https://escholarship.org/uc/item/0tw744hb)
- Revisions that add missing connections help most when readers lack inference skills, domain knowledge or motivation (Britton/Gulgoz follow-up review, 1999). — [Benjamins](https://benjamins.com/catalog/dd.1.2.04gul)
- PoTeC (75 participants, beginner vs advanced physics/biology students, 12 texts). A 2023 analysis found classical eye-movement measures did not reliably predict comprehension, while gaze-recurrence measures did. Experts showed more irregular eye-movement dynamics linked to better comprehension. — [arXiv](https://arxiv.org/abs/2403.00506); [search summary](https://www.zora.uzh.ch/entities/publication/f4970f70-7c79-4291-a2c8-cf9deafe7ea7)
- Buljan 2018 included doctors (n=64) alongside students and consumers, and the format effect (none) did not differ across groups. — [Liverpool repository](https://livrepository.liverpool.ac.uk/id/eprint/3025011)

### Inferences
- Reader background has to be controlled or measured in any human validation. The "intended reader" of a research paper is a domain-adjacent expert, and such readers may show less or reversed sensitivity to cohesion edits.
- An LLM reader's large parametric knowledge makes it behave like a high-knowledge reader. That could explain why it does not respond to cohesion or jargon manipulations, consistent with the reverse cohesion effect. This is an inference.

### Gaps
- Comprehension accuracy by expertise from PoTeC was not extracted (full paper not opened).

## 5. Cost and design of small human studies on paper comprehension

### Takeaway
Feasible designs use about 100–200 readers, each reading 2–3 short texts or one longer one, with multiple-choice questions. QuALITY-style annotation costs about $15–21/hr of reader time, or about $9 per fully validated question. Prolific and CloudResearch give the best cost per high-quality respondent among general platforms (about $2 per respondent for short surveys in Douglas et al. 2023). Expert or graduate-level readers will cost much more; no direct expert-panel pricing was found.

### Cited Findings
- Freeling 2021: 170 students, about 30-minute sessions, up to 3 abstracts each, 5 minutes per abstract. A power analysis from a pilot targeted 600 understanding observations (power >0.99). — [Frontiers](https://www.frontiersin.org/articles/10.3389/fpsyg.2021.714321/full)
- Buljan 2018: n = 64–171 per arm-set. Kerwer 2021: about 2,200 per study for online PLS comparisons. — [Liverpool](https://livrepository.liverpool.ac.uk/id/eprint/3025011); [Frontiers](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.771399/pdf)
- QuALITY: an untimed read of a roughly 5k-token passage plus about 10 questions took 50–60 minutes per worker at about $16/hr. A 45-second-per-question timed pass took 11–12 minutes per task at about $17/hr. Total cost was $9.10 per question. Workers were recruited and vetted via Upwork-style qualification (45 annotators). — [QuALITY](https://aclanthology.org/2022.naacl-main.391.pdf)
- Douglas, Ewell and Brauer 2023 (PLOS ONE) compared MTurk, Prolific, CloudResearch, Qualtrics and SONA:
  - Prolific and CloudResearch had the best attention-check pass rates, instruction following and recall.
  - Cost per high-quality respondent: Prolific $1.90, CloudResearch $2.00, MTurk $4.36, Qualtrics $8.17.
  - Prolific's own blog cites recall accuracy of 83.47% (Prolific) vs 52.20% (MTurk); that is vendor material.
  — [PubMed](https://pubmed.ncbi.nlm.nih.gov/36917576); [Prolific blog](https://www.prolific.com/resources/comparing-data-quality-in-online-research-platforms-why-prolific-outperforms-the-competition)
- OneStop: 360 in-lab participants, each reading 10 articles. In-lab eye-tracking at this scale is a large investment; the paper's compensation details were not retrieved. — [OneStop site](https://lacclab.github.io/OneStop-Eye-Movements)

### Inferences
- Rough budget, extrapolated from QuALITY rates (not quoted by any source). Reading a full research paper (about 8–12k words) and answering 15–20 questions likely takes graduate-level readers 60–120 minutes. At $20–50/hr for screened graduates, that is $40–100 per reader-paper. 10 papers × 2 versions × 15 readers ≈ 300 reader-papers ≈ $12k–30k.
- Effects on accuracy are small (section 2), so within-reader designs, more questions per text, and timing and confidence measures will matter more than adding readers.

### Gaps
- Prolific pricing and pool sizes for graduate-degree or domain-expert screeners were not retrieved.
- No source gives typical time-to-read for a full research paper under comprehension-test conditions.
