# Run checks: runs\kaggle_probe_01\results

- mode probe, commit fbc26a1d25, gpus Tesla T4, 580.178.04, 15360 MiB
Tesla T4, 580.178.04, 15360 MiB
- packages: accelerate                            1.15.0
huggingface_hub                       1.33.0
numpy                                 2.1.3
pillow                                13.0.0
safetensors                           0.8.0
tokenizers                            0.23.3
torch                                 2.11.0+cu130
torchvision                           0.26.0+cu130
transformers                          5.22.1

| step | exit codes | minutes | chunks |
|---|---|---|---|
| fetch data | [0] | 0.22 |  |
| fetch qwen4b | [0] | 0.63 |  |
| qwen4b notes:objects | [0, 0] | 1.02 | 2/2 |
| qwen4b swap:objects | [0, 0] | 1.16 | 2/2 |

## Notes qwen4b objects: 8 cluttered images
- untouched patches whose patch embedding changed (must be 0): 0
- patches where png and processor disagree (must be 0): 0
- largest difference between two runs of the same clean picture (0 means exact): 0.0
- non finite values (must be 0): 0
- mean leak on object squares at the hand over x 1000: 95.381

## Swap qwen4b objects: 8 cluttered images
- largest logit gap A vs plain cluttered: 0.0
- largest logit gap D vs plain clean: 0.0
- A answers equal plain cluttered answers: 100.00%
- D answers equal plain clean answers: 100.00%
- parsed answers: 100.00%
- greedy path equals gold agrees with the scorer (plain rows): 100.00%

| version | images | our accuracy % | study 3 % |
|---|---|---|---|
| clean | 2 | 100.0 | 97.2 |
| random_objects | 2 | 100.0 | 90.0 |
| lookalikes | 2 | 50.0 | 31.2 |
| text_traps | 2 | 50.0 | 57.0 |
| everything | 2 | 0.0 | 12.6 |

