# Datasets for the error-location test beyond maths

Shree's question is whether RL strengthens a model's use of error-location information to guide repairs. Her test uses reasoning traces of four to eight steps with one inserted error, in matched versions with the mistake at different steps, and compares a starting model with its SFT and RL versions. It runs on arithmetic traces, ProcessBench and BIG-Bench Mistake. This page lists datasets outside maths where the same test can run, with recipes and other step-based tasks as the target.

A dataset fits when an error can be inserted at a chosen step, or the first wrong step is labelled, and a script can check the repair. The best fits give the exact state after every step, as arithmetic does.

Checked on 14 Nov 2026 against each paper, and against the data files for ProcBench, CaT-Bench and BIG-Bench Mistake. "Unverified" marks what could not be confirmed.

## Exact state after every step

Any step can hold the error, matched versions are easy to make, and a script checks the repair exactly.

| Dataset | Domain | Size | Steps per problem | The state after each step | Licence | Link |
|---|---|---|---|---|---|---|
| SCONE Alchemy (Long et al., ACL 2016) | beakers of coloured liquid: pour, drain, mix | 4,560 examples (3,661 train, 899 test) | 5 | the contents of every beaker | unverified | nlp.stanford.edu/projects/scone (did not load on 14 Nov) |
| SCONE Scene and Tangrams (same paper) | people and hats in a row of positions; a row of figures | 4,402 and 4,989 examples | 5 | positions of every person or figure | unverified | as above |
| Entity tracking (Kim and Schuster, ACL 2023) | objects moved between boxes | 2,200 generated scenarios | 12 operations | the contents of every box | not stated in the repository | github.com/sebschu/entity-tracking-lms |
| PrOntoQA (Saparov and He, ICLR 2023) | first-order logic in made-up worlds | generated, any number | set by the number of hops | each step is one deduction that parses into a proof step | Apache-2.0 | github.com/asaparov/prontoqa |
| ProofWriter (Tafjord et al., Findings of ACL 2021) | rule-based logic in natural language | 100k questions in each of five sets (D0, D1, D2, D3, D5) | proof depth 0 to 5 | each proof step's conclusion | unverified | allenai.org/data/proofwriter |
| CRUXEval (Gu et al., 2024) | Python | 800 functions | 3 to 13 lines | variable values after each line, from running the function with a tracer | MIT | github.com/facebookresearch/cruxeval |
| ProcBench (Fujisawa et al., 2024) | procedures on strings, lists and numbers, 23 tasks | 5,520 problems; 1,150 have 4 to 8 steps | 2 to 25 | given for every step | CC-BY-4.0 | huggingface.co/datasets/ifujisawa/procbench |
| PlanBench Blocksworld (Valmeekam et al., NeurIPS 2023) | planning | 600 instances | unverified | the world state after each action, from the PDDL domain | MIT | github.com/karthikv792/LLMs-Planning |
| ACPBench (Kokel et al., AAAI) | planning in 13 PDDL domains | 7 question types | single-step questions; traces come from the domains | the domains define every state; one question type asks for the state after an action | not asserted on GitHub | github.com/IBM/ACPBench |
| TextWorld cooking games (Côté et al., 2018) | text games where the player follows a recipe | generated, any number | set by the recipe | the game engine tracks every object | see the repository | github.com/microsoft/TextWorld |

## Recipes with a label at every step

| Dataset | Domain | Size | Steps per recipe | What each step is labelled with | Checking the repair | Licence | Link |
|---|---|---|---|---|---|---|---|
| PizzaCommonSense (Diallo et al., Findings of EMNLP 2024) | pizza recipes from Recipe1M | 1,087 recipes, 13,141 annotated steps | 12.5 on average, median 11 | the input, the cooking action and the output | free text: match against the gold output, or use a judge model | not stated in the repository | github.com/adiallo07/PizzaCommonsense |
| CaT-Bench (Lal et al., 2024) | cooking recipes | 288 recipes in the Hugging Face files; 110 have 4 to 8 steps | median 9 | which earlier steps each step depends on; 4,581 pairs are dependent | a misplaced step breaks a labelled dependency | MIT on Hugging Face | huggingface.co/datasets/vanyacohen/CaT-Bench |
| CREPE (Zhang et al., Findings of EACL 2023) | 183 hand-written procedures, 43 of them recipes | 1,219 steps | 6.7 on average | whether each event became more, less or equally likely; entity states | label match | MIT | github.com/zharry29/causal_reasoning_of_entities_and_events |
| ProPara (Dalvi Mishra et al., NAACL 2018) | science processes | 488 paragraphs about 183 processes | about 6.8 | the existence and location of each entity | match with the labelled grid | Apache-2.0 | github.com/allenai/propara |

## A labelled first wrong step in a model's own trace

These hold errors where a model made them, so the location is fixed and matched versions are not possible. They test the natural-mistake part of the plan.

| Dataset | Domain | Size | Labels | Licence | Link |
|---|---|---|---|---|---|
| REVEAL (Jacovi et al., ACL 2024) | open-domain question answering | 704 questions, 1,002 chains, 3,360 steps | relevance, evidence attribution and logical correctness of every step | CC-BY-ND-4.0 (no adapted copies may be shared) | huggingface.co/datasets/google/reveal |
| DeltaBench (He et al., 2025) | maths, programming, physics, chemistry, biology, general reasoning | 1,236 long chains from o1-style models | sections of each chain labelled by people, including errors | unverified | the paper's GitHub link has moved |
| Mr-Ben (Zeng et al., 2024) | physics, chemistry, logic, coding and more | 5,975 questions | the first error step of a model's solution | MIT | huggingface.co/datasets/Randolphzeng/Mr-Ben |
| VersaPRM data (Zeng et al., 2025) | MMLU-Pro subjects such as law, biology and philosophy | unverified | steps labelled by an automatic method | MIT | github.com/UW-Madison-Lee-Lab/VersaPRM |
| Who&When (Zhang et al., 2025) | multi-agent systems | failure logs from 127 systems | the agent and the decisive error step | unverified | see the paper |
| BIG-Bench Mistake (Tyen et al., ACL 2024), already in the plan | word sorting, tracking, logical deduction, arithmetic, Dyck languages | 2,186 traces; 195 outside arithmetic have the first mistake at step 2 | PaLM 2's first mistake, marked by annotators | Apache-2.0 | github.com/WHGTyen/BIG-Bench-Mistake |

## Where to start

1. **SCONE Alchemy, for recipes.** Every problem is five instructions that pour and mix liquids, with the exact contents of every beaker after each one. That is a recipe with a state a script can check, in exactly the four-to-eight-step range. The data page did not load on 14 Nov, so the download and the licence still need checking.
2. **PizzaCommonSense, for real recipes.** It labels the output of every step of 1,087 real recipes. A wrong output can go at any step. The repair check needs text matching or a judge, since the labels are free text.
3. **PrOntoQA and CRUXEval, for logic and code.** Both give an exact state at every step, PrOntoQA through its proof and CRUXEval through execution. PrOntoQA's generator sets the number of steps, so the four-to-eight-step range is easy to hit.
4. **ProcBench and entity tracking, for state tracking.**
5. **REVEAL or DeltaBench, for the model's own mistakes outside maths.**

## Caveats

- Code is inside the domains Olmo 3's RL stage trained on (general chat, maths and code, per the Olmo 3 paper). CRUXEval tests the effect inside RL's training domains; recipes, logic and state tracking test it outside them.
- REVEAL's licence forbids sharing modified copies, so planted versions can be used but not published.
- BIG-Bench Mistake's `mistake_index` is 0-indexed, and no trace has its first mistake at step 1.
- The CaT-Bench repository states no licence; the Hugging Face card says MIT.
- Recipe mistakes also exist as video (CaptainCook4D, PIE-V), which a text model cannot use.

## Search record

On 14 Nov 2026: web, arXiv and dataset-card searches, two Asta paper searches, and a second pass aimed at dataset families with an exact state at every step (state tracking, logic with proofs, code execution, planning, text games, recipe annotation). Every entry was checked against its paper; sizes for ProcBench, CaT-Bench and BIG-Bench Mistake also against the data files. The Asta search with this test's description returned 24 papers, of which only ProcBench fits.

`docs/recovery-test-beyond-math.md` describes an earlier test on ProcBench and BIG-Bench Mistake that scores final answers after a wrong step 2. It does not vary the error's location.
