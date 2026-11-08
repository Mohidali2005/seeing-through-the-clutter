# Run checks: runs\kaggle_smoke_01\results

- mode smoke, commit 235b679f95, gpus Tesla T4, 580.178.04, 15360 MiB
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
| fetch data | [0] | 0.28 |  |
| fetch qwen4b | [0] | 0.58 |  |
| qwen4b notes:objects | [0, 0] | 1.52 | 2/2 |
| qwen4b swap:objects | [0, 0] | 4.4 | 2/2 |
| qwen4b notes:shapes | [0, 0] | 0.49 | 2/2 |
| qwen4b notes:hard | [0, 0] | 0.7 | 2/2 |
| qwen4b swap:shapes | [0, 0] | 2.05 | 2/2 |
| qwen4b swap:hard | [0, 0] | 5.56 | 2/2 |
| fetch qwen8b | [0] | 1.23 |  |
| qwen8b notes:objects | [1] | 1.03 | 0/2 |
| qwen8b swap:objects | [0] | 12.18 | 2/2 |

## Notes qwen4b hard: 90 cluttered images
- untouched patches whose patch embedding changed (must be 0): 0
- patches where png and processor disagree (must be 0): 0
- largest difference between two runs of the same clean picture (0 means exact): 0.0
- non finite values (must be 0): 0
- mean leak on object squares at the hand over x 1000: 48.620

## Notes qwen4b objects: 40 cluttered images
- untouched patches whose patch embedding changed (must be 0): 0
- patches where png and processor disagree (must be 0): 0
- largest difference between two runs of the same clean picture (0 means exact): 0.0
- non finite values (must be 0): 0
- mean leak on object squares at the hand over x 1000: 74.345

## Notes qwen4b shapes: 30 cluttered images
- untouched patches whose patch embedding changed (must be 0): 0
- patches where png and processor disagree (must be 0): 0
- largest difference between two runs of the same clean picture (0 means exact): 0.0
- non finite values (must be 0): 0
- mean leak on object squares at the hand over x 1000: 21.416

## Swap qwen4b hard: 90 cluttered images
- largest logit gap A vs plain cluttered: 0.0
- largest logit gap D vs plain clean: 0.0
- A answers equal plain cluttered answers: 100.00%
- D answers equal plain clean answers: 100.00%
- parsed answers: 100.00%
- greedy path equals gold agrees with the scorer (plain rows): 100.00%

| version | images | our accuracy % | study 3 % |
|---|---|---|---|
| clean | 10 | 40.0 | 42.2 |
| wrong_color | 10 | 0.0 | 8.4 |
| lookalikes | 10 | 0.0 | 3.2 |
| text_traps | 10 | 30.0 | 28.0 |
| everything | 10 | 0.0 | 0.4 |
| squiggles_2 | 10 | 50.0 | 36.4 |
| squiggles_4 | 10 | 40.0 | 31.6 |
| squiggles_8 | 10 | 20.0 | 26.0 |
| squiggles_16 | 10 | 30.0 | 30.4 |
| other_shapes | 10 | 10.0 | 17.4 |

## Swap qwen4b objects: 40 cluttered images
- largest logit gap A vs plain cluttered: 0.0
- largest logit gap D vs plain clean: 0.0
- A answers equal plain cluttered answers: 100.00%
- D answers equal plain clean answers: 100.00%
- parsed answers: 100.00%
- greedy path equals gold agrees with the scorer (plain rows): 100.00%

| version | images | our accuracy % | study 3 % |
|---|---|---|---|
| clean | 10 | 100.0 | 97.2 |
| random_objects | 10 | 100.0 | 90.0 |
| lookalikes | 10 | 20.0 | 31.2 |
| text_traps | 10 | 60.0 | 57.0 |
| everything | 10 | 10.0 | 12.6 |

## Swap qwen4b shapes: 30 cluttered images
- largest logit gap A vs plain cluttered: 0.0
- largest logit gap D vs plain clean: 0.0
- A answers equal plain cluttered answers: 100.00%
- D answers equal plain clean answers: 100.00%
- parsed answers: 100.00%
- greedy path equals gold agrees with the scorer (plain rows): 100.00%

| version | images | our accuracy % | study 3 % |
|---|---|---|---|
| clean | 10 | 50.0 | 45.0 |
| squiggles | 10 | 30.0 | 28.8 |
| blobs | 10 | 0.0 | 2.8 |
| lookalikes | 10 | 0.0 | 3.4 |

## Swap qwen8b objects: 40 cluttered images
- largest logit gap A vs plain cluttered: 0.0
- largest logit gap D vs plain clean: 0.0
- A answers equal plain cluttered answers: 100.00%
- D answers equal plain clean answers: 100.00%
- parsed answers: 99.77%
- greedy path equals gold agrees with the scorer (plain rows): 100.00%

| version | images | our accuracy % | study 3 % |
|---|---|---|---|
| clean | 10 | 100.0 | 97.2 |
| random_objects | 10 | 90.0 | 90.0 |
| lookalikes | 10 | 30.0 | 31.2 |
| text_traps | 10 | 60.0 | 57.0 |
| everything | 10 | 20.0 | 12.6 |

