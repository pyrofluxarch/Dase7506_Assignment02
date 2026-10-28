# MP2 Final Experiment Report

## Goal and constraints

The task is to choose one of four candidate solutions for each math question. The official scorer evaluates the selected candidate's final numeric answer. The solution uses the supplied fixed candidate embeddings and text-derived features, with no external model or network access during inference. The submission must stay under the 16 MiB inference asset limit and 30-second CPU budget.

## Development method

The development model is a lightweight linear candidate ranker. It combines 1,024 standardized fixed-representation features with 81 handcrafted features for candidate text format, numeric answers, answer agreement, candidate embedding relations, question-number overlap, and simple arithmetic equation consistency. Training uses a groupwise listwise objective plus a pairwise hinge loss and focuses on informative training groups where parsed candidate answers disagree.

During development, multiple models were compared on the training and validation splits. The final training schedule was fixed at 26 epochs per seed based on development experiments. After that selection, the training and validation examples were combined to refit three models (seeds 17, 42, and 123). Their within-question scores are standardized and averaged at inference. Test references are not read by `train_final_refit.py` and were not used to choose features, seeds, epochs, or model weights.

## Final evaluation

The frozen ensemble was run once with the official test prediction and scoring tools:

- Test correct: **393 / 467**
- Test accuracy: **84.1542%**
- Required minimum: 84% (at least 393 / 467)
- Prediction time reported by the runner: **0.955 seconds** in this environment
- Scorer time: **0.024 seconds** in this environment
- Checkpoint size: **22,234 bytes**, far below the 16 MiB limit

Runtime measurements are from the current environment and are not a guarantee of timing on the course's reference machine.

## Limitations and reproducibility

The 84.1542% result is the score of the included frozen checkpoint and exact prediction file under the supplied scorer. The validation metrics from the earlier development checkpoint should not be presented as validation metrics for the refit model, because validation examples were included in final refit after model selection. To reproduce final training, run from the `code` directory:

```bash
python -m pip install -r requirements-train.txt
python train_final_refit.py --output runs/model_bundle_final --epochs 26 --seeds 17 42 123
```

To evaluate the included checkpoint without retraining:

```bash
python predict.py --method submission.py --checkpoint runs/model_bundle_final --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

No test labels or test references are used by the training script. The test score is reported only as the final evaluation of the frozen model.


## Important assignment-rule note

The included 84.1542% checkpoint was refit using both training and validation references after model selection. The course GUIDE states that fitting must use only the supplied training references and validation is for development/selection. Thus, the score is a reproduction result for this checkpoint, but the train+validation refit procedure is not compliant with that stated rule. A fully compliant final submission must be retrained using training references only and re-evaluated; its test accuracy may differ.
