# InterLat

InterLat represents intermediate arithmetic values as editable nodes in a directed acyclic graph. Its numerical channel learns decimal place values and operator coefficients. Editing a node triggers recomputation of its downstream dependents.

## Reproduce the arithmetic experiment

Run from the repository root with Python and NumPy. The release includes the frozen input features, calibrations and protocol. The following commands fit the numerical channel and score each edit with an independent arithmetic reference.

```sh
python -m pip install -r requirements.txt
python "experiment_data/runs/arithmetic_workspace/iterate.py" formal
python "experiment_data/runs/arithmetic_workspace/verify_results.py"
```

The verifier writes `verified.json` next to the program. Each of three fits evaluates the same 10,800 edits. The reported counts cover exact answers, local versus full recomputation, and preservation of unaffected nodes.

## Other experiments

The `interlat` package contains the graph, edit, training and evaluation code. The subdirectories in `experiment_data/runs` are organized by experiment. Run model-based experiments from the repository root with that directory on `PYTHONPATH`. Public model weights, model-training checkpoints and the Belief-R dataset are separate inputs. The GPU experiments use PyTorch and Transformers. The belief evaluation also uses Hugging Face Datasets. Matplotlib is used for plotting.

| Directory | Experiment |
| --- | --- |
| `arithmetic_workspace` | Frozen arithmetic evaluation, ablations, update latency and procurement |
| `neural_baselines`, `belief_fusion` | Neural belief revision and confidence-based fusion |
| `digit_representations`, `quotient_comparisons`, `surface_learning` | Digit representation, comparison errors and transfer to new input forms |
| `end_to_end_latency`, `width_scaling`, `batch_invariance`, `shared_edit_errors` | Latency, graph width, batch composition and shared-input edits |
| `hotel_replay`, `hotel_few_shot`, `hotel_baselines` | Sequential hotel amendments and matched language-model baselines |
| `procurement_few_shot` | Paired procurement decisions with few-shot demonstrations |

For `hotel_baselines`, run `e8.py --help` in its directory to view the model, decoding modes, batch size and output arguments. Its configuration records the checksums of the evaluation script, examples, frozen features and calibration. Hotel cases contain dialogue excerpts from MultiWOZ 2.2. Its copyright notice and license accompany the fixtures. For the frozen procurement script, the original protocol resolves the model at `experiment_data/interlat-code/experiment_data/artifacts/Qwen3-1.7B`; point that location at the same model directory used by the other experiments.

The arithmetic replay uses the archived features and calibrations. The auxiliary curriculum generators, training baselines and neural answer reader expose English prompts in this edition. For exact replay of those earlier neural measurements, the preceding repository revision contains the original-language templates.
