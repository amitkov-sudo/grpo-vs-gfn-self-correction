# Datasets for error-location recovery beyond maths

Shree's planted-error results (OLMo 3 7B and 32B, SFT, DPO and RL checkpoints, 50 MATH problems with a wrong step 2) raise the question of what else they generalize to. A dataset fits when its answers are sequences of steps, an error can sit at a known step, and recovery can be scored without a person.

Searched on 14 Nov 2026. Facts come from each paper, dataset card or repository. Entries marked "unverified" could not be confirmed.

| Dataset | Domain | Size | Error location | Automatic scoring | Licence | Link |
|---|---|---|---|---|---|---|
| BIG-Bench Mistake (Tyen et al., ACL 2024) | Word sorting, tracking shuffled objects, logical deduction, multistep arithmetic, Dyck languages | 2,186 traces | First mistake annotated per trace | Exact match with the target | Apache-2.0 | github.com/WHGTyen/BIG-Bench-Mistake |
| PlanBench, Blocksworld (Valmeekam et al., 2022) | Planning | 600 Blocksworld instances | Planted by script: replace action 2 of the reference plan | The VAL plan validator checks the goal state | MIT (repository); the Hugging Face mirror states none | github.com/karthikv792/LLMs-Planning |
| Mr-Ben (Zeng et al., 2024) | Physics, chemistry, logic, coding and more | 5,975 questions | First error step annotated | Exact match on the answer; coding items unverified | MIT | huggingface.co/datasets/Randolphzeng/Mr-Ben |
| HumanEvalFix, in HumanEvalPack (Muennighoff et al., 2023) | Code, six languages | 164 per language | Buggy solutions with a bug type; a script can plant a bug at line 2 | Unit tests | MIT | huggingface.co/datasets/bigcode/humanevalpack |
| TRIP (Storks et al., 2021) | Stories of physical actions | Unverified | The conflicting sentence pair is annotated | Label match, not recovery | Not stated | github.com/sled-group/Verifiable-Coherent-NLU |
| CaT-Bench (Lal et al., 2024) | Recipe plans | 9,162 questions on 300 plans | Step dependencies annotated; a script could swap two dependent steps | Yes or no labels, not recovery | MIT on Hugging Face | huggingface.co/datasets/vanyacohen/CaT-Bench |

## Three to start with

1. **BIG-Bench Mistake.** Keep the traces whose first mistake is at step 2, or plant one by script (word sorting and arithmetic are the easiest). Score the final answer by exact match. Tsui (2026) used two of its tasks for the only non-maths results in Self-Correction Bench.
2. **PlanBench Blocksworld.** Replace action 2 of the reference plan with a legal but wrong action, ask the model to finish the plan, and score it with the VAL validator. Keep the planted action reversible so that recovery stays possible.
3. **Mr-Ben.** Truncate a solution at its annotated first error step and score the answer by exact match. Check its subject mix before relying on it.

## Caveats

- Recipes, the domain Shree mentioned, have no dataset here that scores recovery automatically. CaT-Bench and TRIP score labels only. A recipe version would need a planted step and a judge model.
- HumanEvalFix is small, at 164 problems per language. Check how many BIG-Bench Mistake traces have their first mistake at step 2.
- TRIP and the CaT-Bench repository state no licence. REALMistake and FOLIO are gated, and CLUTRR's licence is unclear.
