input = """
Going to sample between 1 and 4 traces per predictor.
Will attempt to bootstrap 16 candidate sets.
Processing test data:   0%|                      | 0/100 [00:00<?, ?it/s]/Users/hazn/Desktop/code.nosync/peopen/.venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
0.7903566360473633
Processing test data:   1%|▏             | 1/100 [00:00<01:13,  1.35it/s]0.7806801795959473
Processing test data:   2%|▎             | 2/100 [00:00<00:35,  2.74it/s]0.7276570200920105
0.724469780921936
Processing test data:   4%|▌             | 4/100 [00:00<00:17,  5.52it/s]
0.6953457593917847
Processing test data:   5%|▋             | 5/100 [00:01<00:15,  6.25it/s]0.7455343008041382
Processing test data:   6%|▊             | 6/100 [00:01<00:13,  6.85it/s]0.7276465892791748
0.7667615413665771
Processing test data:   8%|█             | 8/100 [00:01<00:10,  8.82it/s]0.7029622793197632
0.6977053880691528
Processing test data:  10%|█▎           | 10/100 [00:01<00:10,  8.26it/s]0.7492201328277588
0.7259599566459656
Processing test data:  12%|█▌           | 12/100 [00:01<00:09,  9.33it/s]0.6222596168518066
0.7348186373710632
Processing test data:  14%|█▊           | 14/100 [00:02<00:09,  8.87it/s]
0.7011666297912598
Processing test data:  15%|█▉           | 15/100 [01:47<29:22, 20.73s/it]
0.7357615232467651
Processing test data:  16%|█▊         | 16/100 [04:51<1:19:14, 56.60s/it]
0.7193813323974609
Processing test data:  17%|█▊         | 17/100 [04:56<1:01:10, 44.22s/it]
0.7179163098335266
Processing test data:  18%|█▉         | 18/100 [07:22<1:35:45, 70.06s/it]
0.6633777618408203
Processing test data:  19%|█▉        | 19/100 [10:35<2:18:40, 102.72s/it]
0.729771077632904
Processing test data:  20%|██        | 20/100 [13:33<2:44:47, 123.59s/it]
0.724798321723938
Processing test data:  21%|██        | 21/100 [14:41<2:22:12, 108.01s/it]
0.7190718054771423
Processing test data:  22%|██▏       | 22/100 [16:30<2:20:41, 108.22s/it]
0.7438332438468933
Processing test data:  23%|██▎       | 23/100 [18:07<2:14:26, 104.76s/it]
0.6872397661209106
Processing test data:  24%|██▍       | 24/100 [20:38<2:30:07, 118.52s/it]
0.7119857668876648
Processing test data:  25%|██▌       | 25/100 [24:55<3:19:17, 159.44s/it]
0.7660192847251892
Processing test data:  26%|██▌       | 26/100 [26:42<2:57:34, 143.98s/it]
0.6771209836006165
Processing test data:  27%|██▋       | 27/100 [27:55<2:29:19, 122.73s/it]
0.7119537591934204
Processing test data:  28%|██▊       | 28/100 [31:26<2:58:45, 148.97s/it]
0.6904003620147705
Processing test data:  29%|██▉       | 29/100 [35:52<3:37:37, 183.90s/it]
0.7006536722183228
Processing test data:  30%|███       | 30/100 [40:28<4:06:46, 211.52s/it]
0.6751941442489624
Processing test data:  31%|███       | 31/100 [40:28<2:50:26, 148.21s/it]
0.7478059530258179
Processing test data:  32%|███▏      | 32/100 [45:14<3:34:58, 189.68s/it]
0.7005113959312439
Processing test data:  33%|███▎      | 33/100 [46:26<2:52:05, 154.11s/it]
0.7179872989654541
Processing test data:  34%|███▍      | 34/100 [48:03<2:30:50, 137.12s/it]
0.7219803929328918
Processing test data:  35%|███▌      | 35/100 [49:26<2:11:05, 121.00s/it]
0.6292373538017273
Processing test data:  36%|███▌      | 36/100 [52:42<2:33:01, 143.46s/it]
0.7767099142074585
Processing test data:  37%|███▋      | 37/100 [56:15<2:52:30, 164.29s/it]
0.7490599751472473
Processing test data:  38%|███▊      | 38/100 [58:17<2:36:41, 151.63s/it]
0.7030396461486816
Processing test data:  39%|███     | 39/100 [1:00:21<2:25:37, 143.25s/it]
0.7132048606872559
Processing test data:  40%|███▏    | 40/100 [1:03:04<2:29:13, 149.22s/it]
0.6771841645240784
Processing test data:  41%|███▎    | 41/100 [1:03:49<1:56:00, 117.98s/it]
0.6937202215194702
Processing test data:  42%|███▎    | 42/100 [1:06:15<2:01:59, 126.20s/it]
"""

# average every line that starts with 0.

import re
import numpy as np

lines = input.split("\n")
lines = [line for line in lines if line.startswith("0")]
lines = [re.findall(r"0\.\d+", line) for line in lines]
lines = [float(line[0]) for line in lines]
print(np.mean(lines))  # 0.7154715606144496
print(np.std(lines))  # 0.03252856618123365
