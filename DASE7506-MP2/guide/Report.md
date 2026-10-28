# MP2 Report: Selecting Reliable LLM Reasoning

**Course:** DASE7506 — Advanced Machine Learning Methods, Fall 2026  
**Assignment:** MP2 — Selecting Reliable LLM Reasoning  
**Submission type:** Train-only compliant candidate-selection system  
**Protocol:** `dase7506-mp2-open-hard-verifier-v5`  
**Status:** Reproducible baseline; the compliant model does **not** reach the aspirational 84% test-accuracy target.

> **Result integrity note.** This report documents the model and scores recorded in this package. The frozen train-only checkpoint scores **384 / 467 = 82.22698%** on the supplied test split and **750 / 798 = 93.98496%** on validation. An earlier experimental checkpoint scored 84.1542% after fitting on training and validation references; it is excluded because that fitting procedure does not comply with the train-only fitting rule. Do not attribute that earlier score to this package.

## Executive Summary

The task is to choose the most reliable one of four candidate mathematical solutions for each question. The language model has already generated the candidates, and the assignment supplies fixed 1,024-dimensional representations for the candidates. The submission therefore does not generate new reasoning or call an external language model. Instead, it ranks the four candidates using the provided representations and inexpensive features extracted from the question and candidate texts.

The final implementation is a small linear ranking ensemble. It concatenates standardized candidate representations with 81 handcrafted features, then scores candidates with two independently trained linear rankers. The handcrafted features summarize answer parsing and formatting, numeric content, answer agreement among the four candidates, relationships between candidate answers, candidate/question number overlap, and simple equation-to-answer consistency. Training combines a groupwise listwise ranking loss with a pairwise hinge loss. The two model members use random seeds 17 and 42; their within-question scores are standardized and averaged at inference.

The model is intentionally small. The recorded checkpoint is approximately 18 KiB, far below the 16 MiB inference-asset limit. In the recorded environment, prediction took approximately 0.97 seconds and the independent scorer approximately 0.024 seconds; these measurements are machine-dependent and are not a guarantee for the course reference machine. The official test result is 82.22698%, which is below the desired 84% threshold. This report treats that shortfall honestly and discusses what the evidence does and does not establish.

The central conclusion is that a compact ranker is a practical baseline under the compute and asset constraints, but the validation-to-test gap suggests that the current features and learned decision boundary do not generalize equally well to the harder test distribution. More elaborate models are not automatically better; future work should focus on error analysis, stronger mathematical consistency checks, careful grouped validation, and ablations that isolate which information actually improves selection.

## 1. Problem Definition and Research Questions

### 1.1 Task formulation

For each question, the dataset contains four candidate solutions and a fixed vector representation for each candidate. The system must output one candidate index. The official scorer evaluates whether the selected candidate's parsed final numeric answer matches the reference answer. This metric is not a full proof checker: a candidate may contain flawed reasoning but still end with the correct numeric answer, while a polished explanation may reach an incorrect result.

The task is consequently best framed as **within-question candidate ranking**. Given question group \(q\), candidates \(c_1, c_2, c_3, c_4\), and feature representation \(x_{qj}\), the model computes a score \(s_{qj}\) for each candidate and returns \(\arg\max_j s_{qj}\). Scores only need to be meaningfully comparable within the four candidates for the same question.

### 1.2 Research questions

The assignment encourages investigation of several issues:

1. Does the supplied representation help select the correct solution?
2. Do text, answer-format, and numeric consistency features add useful evidence?
3. Does comparing candidates within a question help beyond independent candidate scoring?
4. Can simple consistency checks identify plausible-looking but incorrect solutions?
5. Is an improvement in accuracy worth its inference-time and asset cost?

The implemented model is designed to investigate these questions within a small, CPU-friendly system. The current experiment log does not contain a complete factorial ablation for every feature family, so conclusions about individual features are deliberately limited to the experiments that were actually recorded.

### 1.3 Constraints

The implementation follows the assignment's practical constraints: no large external verifier, no extra pretrained model weights, a maximum of 16 MiB of uncompressed inference assets, and at most 30 seconds for a fresh prediction process plus independent scoring on the complete test bank under the reference machine's four-thread limit. The supplied data and source code are not counted as learned inference assets, but model weights, learned normalization statistics, configuration values, indexes, and ensemble members are.

## 2. Data and Evaluation Protocol

### 2.1 Data splits

The package contains training, validation, and test splits. The training split contains 4,755 questions. The validation split contains 798 questions, and the test split contains 467 questions under the provided evaluation protocol. The exact dataset files and their manifests are included under `code/data/`.

The data protocol is central to the validity of the result:

- **Training:** model parameters and normalization statistics are fitted from training inputs and training references only.
- **Validation:** used during development to compare configurations and select hyperparameters. Validation labels are not loaded by the final fitting script.
- **Test:** used only after predictions are frozen, through the independent official scorer. Test references are not loaded by the training script and are not used for hyperparameter selection.

This separation avoids fitting directly to validation or test labels. It also means the final model cannot use the validation labels as additional supervised examples, even if doing so might improve the score. Compliance takes priority over a score obtained with a disallowed training procedure.

### 2.2 Metric

The official accuracy is

\[
\text{Accuracy} = \frac{\text{number of selected candidates with a correct parsed final answer}}{\text{number of scored questions}}.
\]

On the test split, 384 of 467 selections are correct, giving 82.22698072805139%. To reach at least 84% on 467 questions, the model would need at least 393 correct selections. The current compliant result is nine questions short of that count. A percentage point difference can therefore correspond to a small number of individual questions, and results should be reported with both numerator and denominator.

### 2.3 Why validation and test can differ

Validation accuracy is 750/798 = 93.98496%, much higher than the test accuracy. This gap is a warning that the validation score is not a reliable guarantee of test performance. The two splits have different examples and may differ in difficulty or in the frequency of particular error patterns. The available summary does not establish which exact distribution shift causes the gap. It would be inappropriate to claim a specific cause without an error-by-category analysis.

A further point is that the test set is evaluated under an open-hard verifier protocol. Such a subset can be more challenging than a broad validation set because candidates with differing parseable answers are especially difficult to rank. Future experiments should compare like-for-like difficulty subsets where the protocol allows it and should avoid repeatedly tuning against the held-out test score.

## 3. Method Overview

### 3.1 Design rationale

The assignment supplies candidate representations, so a large generative model is unnecessary for the core prediction step. A compact supervised ranker is appealing because it is fast, auditable, and easy to package. The design combines two sources of evidence:

- **Dense representation evidence:** 1,024 values supplied for each candidate solution.
- **Explicit text and numeric evidence:** 81 handcrafted features derived from the candidate, its parsed final answer, the question, and the other three candidates.

The final feature vector contains 1,105 dimensions: 1,024 representation features plus 81 handcrafted features. This design is a compromise between learning from the supplied semantic representation and exposing task-specific numeric cues directly to the model.

### 3.2 Inference pipeline

For each question, inference follows these steps:

1. Load the four candidate records and their supplied representations.
2. Parse each candidate's final numeric answer using the provided answer-parsing utility.
3. Calculate handcrafted features, including per-candidate features and cross-candidate relations.
4. Standardize the fixed representations and handcrafted features using means and standard deviations stored in the training-only checkpoint.
5. Compute a score for each candidate with each linear ranker.
6. Standardize each ranker's four scores within the question and average the two members' scores.
7. Select the candidate with the highest combined score and write the candidate index in the required prediction format.

No reference answer is needed during prediction. The model uses the candidate text, question text, fixed representations, and learned parameters only. The independent scoring script subsequently compares frozen predictions with the official reference answers.

### 3.3 Linear ranker

For candidate feature vector \(x_{qj}\), each member computes a linear score

\[
s_{qj} = w^\top x_{qj} + b,
\]

where \(w\) and \(b\) are learned from training examples. Linear scoring has limited capacity compared with a large neural verifier, but it offers several advantages for this assignment: low memory use, low inference latency, deterministic feature computation, and relatively straightforward debugging.

The ensemble uses two independently initialized members. For each question, each member's four scores are standardized before averaging. This reduces the risk that one member dominates simply because its raw score scale differs from the other member's scale. The ensemble does not require separate large assets or external model calls.

## 4. Feature Engineering

### 4.1 Fixed candidate representations

The provided 1,024-dimensional representation is the primary dense input. It may encode broad semantic and reasoning information from the candidate solution, but the submission does not assume that it perfectly represents correctness. The values are standardized using training-split statistics and clipped to a bounded range to reduce the effect of extreme values.

The representation is fixed by the assignment, so the model does not backpropagate through the source language model or alter the representation generator. This keeps training and inference inexpensive and makes the method compatible with the provided assets.

### 4.2 Basic text and formatting features

The handcrafted feature set includes inexpensive indicators of candidate structure and answer presentation. Examples include solution length, line or sentence structure, use of answer markers such as `####` or `answer`, and whether a final-answer phrase appears. These signals can be useful because candidates often follow recurring solution templates.

However, formatting is only a proxy for reliability. Longer solutions can contain more useful derivation, but they can also contain more opportunities for error. A neat final-answer marker can make an answer easier to parse without making it mathematically correct. The model therefore uses formatting cues as part of a larger feature vector rather than as a hard-coded rule.

### 4.3 Numeric and final-answer features

Candidate answers are parsed where possible. Numeric features represent properties such as whether an answer can be parsed, whether it is an integer or decimal, whether a fraction-like form is present, and whether the final answer appears in the tail of the solution. The implementation also considers the numeric tokens found in the candidate and the question.

These features are especially relevant because the official metric checks the final numeric answer. The candidate's intermediate reasoning may be long and fluent, but the final answer still has to match the reference. At the same time, parsing can fail on unusual formatting, symbolic expressions, or ambiguous answer markers. Missing or unparsable answers should be treated as uncertainty rather than as proof that a candidate is wrong.

### 4.4 Candidate agreement and relations

The model includes answer-agreement features across the four candidates. If multiple candidates produce the same parseable answer, this agreement can be informative. Other features indicate whether a candidate's answer is unique or whether it differs from the other candidates. Candidate representations are also compared through within-question relations.

Agreement is not a guarantee of correctness. Several candidates may share the same misconception or repeat a common arithmetic mistake. For this reason, agreement features are soft evidence learned from the training set rather than a majority-vote rule. This is a key limitation and a potential avenue for future analysis.

### 4.5 Question-to-answer and equation consistency

The feature code checks simple relations between numeric content in the question and candidate, including whether the final answer matches numeric values appearing in the solution, and whether the right-hand side of an equation agrees with the parsed final answer. A small safe numeric-expression parser is used for a limited set of arithmetic expressions. The intent is to detect internal inconsistencies cheaply.

These are lightweight heuristics, not a general mathematical theorem prover. They cannot reliably understand every unit conversion, word problem condition, algebraic transformation, or semantic relationship. They may miss errors that require interpreting the problem statement, and they may be uncertain when a candidate uses a valid but unconventional derivation. The implementation therefore exposes these signals as features and lets the ranker learn their predictive value.

### 4.6 Standardization and feature scale

The training script calculates feature means and standard deviations from training inputs only. Standard deviations are guarded against near-zero values to avoid division by zero. At inference, the same saved statistics are applied; test-time data is not used to recompute them. The handcrafted feature block is multiplied by a selected scale before concatenation. The chosen feature scale is 1.5 based on development experiments.

This standardization is important because the supplied representation dimensions and handcrafted features can have very different numeric ranges. Without it, a feature could dominate the score simply because of its scale rather than because it contains stronger evidence.

## 5. Learning Objective and Training Procedure

### 5.1 Groupwise ranking

Each training question forms a group of four candidate scores. The listwise component encourages probability mass to concentrate on the candidate or candidates whose parsed answer matches the training reference. Groupwise training aligns with the task because the final decision is made among four candidates for the same question, rather than by independently thresholding each candidate.

### 5.2 Pairwise hinge term

The training objective also contains a pairwise hinge-ranking component. For a correct candidate score \(s^+\) and an incorrect candidate score \(s^-\), a hinge term penalizes cases where the correct score is not larger than the incorrect score by a specified margin. In simplified form,

\[
L_{pair} = \max(0, m - (s^+ - s^-)),
\]

where \(m\) is the margin. The combined loss is the groupwise loss plus a weighted pairwise term. The selected configuration uses pairwise weight 0.25 and margin 0.5. This gives the model an explicit incentive to separate correct and incorrect candidates while retaining the groupwise objective.

Training focuses on informative groups where candidate answers differ and at least one candidate matches the parsed training reference. This avoids spending the same amount of optimization effort on groups that provide little ranking signal. It also makes the effective number of training groups smaller than the total number of training questions; the recorded run used 2,114 informative groups out of 4,755.

### 5.3 Hyperparameters

The selected configuration is:

| Parameter | Value |
|---|---:|
| Training epochs per member | 26 |
| Random seeds | 17, 42 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Pairwise loss weight | 0.25 |
| Pairwise margin | 0.5 |
| Handcrafted feature scale | 1.5 |
| Total feature dimensions | 1,105 |

These values were selected during development by comparing validation results. The final fitting script then trains with the selected settings using training references only. Hyperparameter selection on validation does not mean validation labels are used to fit the final weight vectors; the distinction is documented in the training script and summary.

### 5.4 Reproducibility

The final package records seeds, epoch count, loss configuration, training loss, feature count, checkpoint size, and flags indicating that validation and test data were not loaded for fitting. Fixed seeds make the training procedure more reproducible, although exact numerical results can still vary with library versions, numerical kernels, or environment differences. The supplied checkpoint is the primary artifact for reproducing inference without retraining.

## 6. Experiments and Ablation Evidence

### 6.1 Validation-based model selection

All reported development comparisons use the validation split. The selected train-only configuration achieved 750 correct selections out of 798, or 93.98496%. This was the best validation result among the configurations recorded in the experiment log.

| Experiment | Validation correct | Validation accuracy |
|---|---:|---:|
| Selected pairwise weight 0.25 | 750 / 798 | 93.9850% |
| Pairwise weight 0.0 | 749 / 798 | 93.8596% |
| Pairwise weight 0.5 | 747 / 798 | 93.6090% |
| Learning rate 0.0003 | 746 / 798 | 93.4837% |
| Handcrafted feature scale 0.75 | 747 / 798 | 93.6090% |
| 10 epochs | 744 / 798 | 93.2331% |
| 50 epochs | 746 / 798 | 93.4837% |

The differences are modest. In this set of experiments, the pairwise term with weight 0.25 performed slightly better than removing the pairwise term or increasing its weight to 0.5. A lower learning rate and a different feature scale also did not improve the validation score. Training for longer was not automatically beneficial: 50 epochs scored below 26 epochs, which is consistent with the possibility that longer fitting can overfit the available training examples or move away from a useful optimization point.

These comparisons should not be overinterpreted. They vary one setting at a time in a small experiment set, and the table does not isolate the contribution of every one of the 81 handcrafted features. The evidence supports selecting the recorded configuration among those tested; it does not prove that this configuration is globally optimal.

### 6.2 Test evaluation of the frozen model

After the configuration was selected and the model was fitted on training data only, the frozen checkpoint was evaluated using the independent scorer. The recorded result is:

| Metric | Result |
|---|---:|
| Protocol | `dase7506-mp2-open-hard-verifier-v5` |
| Correct | 384 |
| Questions | 467 |
| Accuracy | 82.22698072805139% |
| Prediction time | Approximately 0.967 seconds in the recorded environment |
| Scoring time | Approximately 0.024 seconds in the recorded environment |
| Checkpoint file size | 18,147 bytes in the packaged checkpoint |

The validation-to-test difference is approximately 11.76 percentage points. This is the most important result to investigate next. The current experiment log does not provide a per-question error taxonomy, so it cannot establish whether the gap is mainly caused by arithmetic mistakes, answer parsing, distribution shift, or another factor. A disciplined next step is to analyze errors on training and validation examples, formulate hypotheses, and validate improvements without using test labels for tuning.

### 6.3 Required but incomplete ablations

The assignment asks for evidence about whether reasoning text, formatting, and candidate comparisons are useful. The current experiment table tests the ranking loss weight, learning rate, feature scale, and epoch count, but does not yet contain a complete ablation suite for all information sources. The following experiments are therefore recommended as future work and are **not claimed as completed**:

1. Fixed representation only.
2. Handcrafted features only.
3. Fixed representation plus handcrafted features.
4. Remove answer-agreement features.
5. Remove length and formatting features.
6. Remove equation-consistency features.
7. Remove within-question candidate-relation features.
8. Compare independent candidate scoring with groupwise ranking.
9. Compare a single seed with the two-seed ensemble.

Each ablation should use the same training split, the same validation split, and a predeclared selection criterion. Test scoring should be reserved for the final frozen model to avoid test-set overfitting.

## 7. Error Analysis and Limitations

### 7.1 What the current evidence suggests

The model has a strong validation score but a substantially lower test score. That pattern suggests a generalization problem, but it does not identify the cause by itself. Possible explanations include differences in difficulty between splits, a mismatch between development features and test errors, or insufficient training examples for certain reasoning patterns. These are hypotheses, not established findings.

### 7.2 Limitations of answer parsing

The official evaluator focuses on the parsed final numeric answer. The model uses the same broad kind of answer extraction to create features, but candidate text can use different formats, multiple numbers, fractions, decimals, or answer markers. If the parsed answer is wrong or missing, downstream features may be misleading. A candidate with an incorrect intermediate calculation but a correct final number may still be counted correct by the official scorer, so reasoning validity and metric correctness are not identical.

### 7.3 Limits of shallow numeric checks

Simple equation checks can detect some contradictions between a written equation and the final answer. They cannot fully evaluate natural-language conditions, units, rates, or multi-step dependencies. For example, in a rate-time-distance problem, a candidate can multiply correctly after choosing the wrong time interval. The arithmetic itself may be internally consistent while the interpretation of the question is wrong. Detecting that kind of mistake requires stronger semantic alignment between the question and the reasoning steps.

### 7.4 Limits of linear ranking

A linear model can combine many useful signals but cannot learn arbitrary interactions unless those interactions are represented explicitly in the features. A rule such as “this answer is plausible only when a particular unit conversion and a specific final-answer pattern occur together” may be difficult to express with the current representation. More complex models could capture richer interactions, but they can also overfit and consume more computation. The current experiments do not show that a more complex model would reliably improve test accuracy.

### 7.5 Candidate agreement can mislead

Multiple candidates may agree because they share the same misconception, or a correct answer may be a minority answer. Agreement should therefore remain a soft learned feature. A majority-vote rule would be too brittle and would ignore whether the candidate's derivation is consistent with the question.

### 7.6 Validation overfitting

Repeatedly trying configurations and selecting the best validation score can gradually overfit the validation split, even if validation data is never used in gradient updates. This is why the number of experiments and the selection process should be documented. A future iteration should use grouped cross-validation within the training split where practical, then reserve validation for final model selection. The official test should remain a one-time or limited final check after freezing the method.

## 8. Resource Use and Compliance

### 8.1 Inference assets

The packaged model file is 18,147 bytes according to the official result package, approximately 0.0173 MiB. This is far below the 16 MiB limit. The asset is a NumPy archive containing linear model weights and learned normalization statistics. The method uses no extra pretrained language model weights or remote API calls.

### 8.2 Runtime

The recorded prediction time is approximately 0.967 seconds, and the independent scoring script takes approximately 0.024 seconds in the recorded environment. The complete measurement is comfortably below the 30-second threshold in that environment. Runtime depends on CPU, Python and NumPy versions, process startup, and file-system performance. The assignment's reference machine is a Xeon Platinum 8457C with a total four-thread budget; the supplied run metadata does not record enough local hardware detail to claim a direct reference-machine timing comparison. The evaluator's official timing should be used for eligibility.

### 8.3 Training and inference separation

Training requires NumPy and the dependencies listed in `requirements-train.txt`. Inference uses the supplied data, the model checkpoint, and the prediction implementation. The prediction script writes a frozen prediction JSON; the independent scorer then validates the protocol and computes accuracy. Keeping these stages separate makes it easier to reproduce the result and reduces the risk of accidentally using labels inside prediction.

### 8.4 Train-only compliance checklist

- The fitting script loads `train` inputs and training references only.
- Feature means and standard deviations are computed from training inputs only.
- Validation is used for development/model selection, not final parameter fitting.
- Test inputs are loaded for prediction after the model is frozen; test references are not loaded by the fitting script.
- Test labels are read only by the independent scorer after predictions have been generated.
- The package excludes the earlier train-plus-validation refit checkpoint.

The user should keep the code and checkpoint versions matched. Replacing `submission.py` or `model.npz` independently can change the behavior and invalidate the recorded result.

## 9. Installation and Reproduction

Run all commands from the `code/` directory. On Windows PowerShell, first change to the directory that contains `predict.py`, `submission.py`, and `train_submission.py`.

### 9.1 Install runtime dependencies

```powershell
python -m pip install -r requirements.txt
python check_data.py
```

The data check should complete without reporting missing or inconsistent required files. If training from scratch, install the training dependencies as well:

```powershell
python -m pip install -r requirements-train.txt
```

### 9.2 Evaluate the included checkpoint without retraining

```powershell
python predict.py --method submission.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

The first command creates `runs/test-predictions.json`. The second command requires that file to exist; running the scorer first will produce a `FileNotFoundError`. The default checkpoint path is `runs/model_bundle`, which must contain `model.npz`.

To evaluate validation predictions instead, use `--split validation` consistently in both commands and choose a different output file name. The protocol and split recorded in the predictions file must match the scorer arguments.

### 9.3 Reproduce training

```powershell
python train_submission.py --output runs/model_bundle --epochs 26 --seeds 17 42
```

This command trains the selected two-member ensemble using training references only and writes `model.npz` plus a training summary. Training again can overwrite the existing checkpoint, so back up the provided checkpoint before reproducing training if you need to preserve the exact recorded artifact.

### 9.4 Output files

- `runs/model_bundle/model.npz`: model weights and training-derived normalization statistics.
- `runs/model_bundle/training_summary.json`: selected configuration and training metadata.
- `runs/model_bundle/validation-score.json`: recorded validation evaluation.
- `runs/model_bundle/test-score.json`: recorded test evaluation for the frozen package.
- `runs/test-predictions.json`: predictions generated by the reproduction command.
- `runs/test-score.json`: score computed from the newly generated predictions.

The `test-score.json` bundled with the checkpoint is a record of a previous run. To independently reproduce the result, run prediction first and then the scoring command, and compare the newly generated output.

## 10. Conclusion and Future Work

This project implements a compact, train-only compliant solution to selecting one answer from four mathematical candidate solutions. It combines supplied fixed representations with explicit numeric, formatting, agreement, and equation-consistency features. A two-seed linear ranker trained with listwise and pairwise ranking objectives offers a practical balance between simplicity, speed, and model size.

The model achieves 93.98496% validation accuracy and 82.22698% test accuracy. It uses an approximately 18 KiB checkpoint and runs quickly in the recorded environment. These results demonstrate that lightweight candidate ranking can be competitive under strict inference constraints, but they also reveal a substantial validation-to-test gap. The current train-only model does not meet the 84% test target, and the report does not claim otherwise.

The highest-priority next steps are: (1) perform a structured error analysis on training and validation examples; (2) complete the requested feature-family ablations; (3) improve question-to-reasoning consistency checks for conditions, units, and arithmetic; (4) compare candidate-group ranking with independent scoring under the same data protocol; and (5) use grouped training-only cross-validation to reduce dependence on a single validation split. Any improvement should be selected using training/validation data and then evaluated on the test split only after the final configuration is frozen.

A useful result is not only a high score but a defensible experiment: the method, data protocol, asset size, timing, checkpoint, and limitations must all be reported honestly. In this submission, the train-only restriction is respected even though the desired 84% threshold has not yet been reached.

## AI Assistance and Reuse Disclosure

AI assistance was used to help draft and revise code, feature ideas, experiment analysis, and this documentation. The student should review every part of the submission, ensure that it reflects the actual implementation and experiments, and disclose AI assistance in the form required by the course. The supplied dataset and starter repository should be credited according to the assignment's attribution instructions. No external pretrained model weights are added by this submission.

## Quick Reference

| Item | Value |
|---|---|
| Model | Two-seed linear listwise + pairwise ranker |
| Seeds | 17, 42 |
| Epochs per seed | 26 |
| Features | 1,024 fixed representation + 81 handcrafted = 1,105 |
| Validation | 750 / 798 = 93.98496% |
| Test | 384 / 467 = 82.22698% |
| Test target | 84% (requires at least 393 / 467) |
| Checkpoint | `runs/model_bundle/model.npz` |
| Checkpoint size | 18,147 bytes recorded in package evaluation |
| Prediction entry point | `predict.py` |
| Scoring entry point | `score_predictions.py` |
| Training entry point | `train_submission.py` |
| Training data rule | Fit model and normalization on training split only |
