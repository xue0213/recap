# Without-Module Ablation Results

## Average Across Target Graphs

| Method | AUROC | AUPRC | Seeds |
|---|---:|---:|---:|
| RECAP | 0.7448 +/- 0.0025 | 0.2658 +/- 0.0057 | 5 |
| RECAP C=1 | 0.7396 +/- 0.0010 | 0.2645 +/- 0.0025 | 5 |
| RECAP w/o Adhesion Score | 0.6173 +/- 0.0129 | 0.1200 +/- 0.0118 | 5 |
| RECAP w/o Context Score | 0.7316 +/- 0.0019 | 0.2615 +/- 0.0054 | 5 |
| RECAP w/o L_H | 0.7443 +/- 0.0026 | 0.2658 +/- 0.0054 | 5 |
| RECAP w/o L_con | 0.7420 +/- 0.0032 | 0.2660 +/- 0.0022 | 5 |
| RECAP w/o Residual | 0.5339 +/- 0.0019 | 0.1060 +/- 0.0005 | 5 |

## Per Dataset

| Method | Dataset | AUROC | AUPRC | Seeds |
|---|---|---:|---:|---:|
| RECAP | ACM | 0.7806 +/- 0.0035 | 0.3719 +/- 0.0035 | 5 |
| RECAP | Amazon | 0.7288 +/- 0.0073 | 0.1321 +/- 0.0039 | 5 |
| RECAP | BlogCatalog | 0.7303 +/- 0.0050 | 0.3281 +/- 0.0024 | 5 |
| RECAP | Facebook | 0.5886 +/- 0.0105 | 0.0471 +/- 0.0131 | 5 |
| RECAP | Reddit | 0.5938 +/- 0.0020 | 0.0469 +/- 0.0006 | 5 |
| RECAP | citeseer | 0.9017 +/- 0.0032 | 0.4548 +/- 0.0159 | 5 |
| RECAP | cora | 0.8213 +/- 0.0032 | 0.4043 +/- 0.0190 | 5 |
| RECAP | weibo | 0.8135 +/- 0.0069 | 0.3415 +/- 0.0043 | 5 |
| RECAP C=1 | ACM | 0.7932 +/- 0.0013 | 0.3754 +/- 0.0021 | 5 |
| RECAP C=1 | Amazon | 0.7411 +/- 0.0083 | 0.1357 +/- 0.0028 | 5 |
| RECAP C=1 | BlogCatalog | 0.7276 +/- 0.0021 | 0.3356 +/- 0.0015 | 5 |
| RECAP C=1 | Facebook | 0.5826 +/- 0.0071 | 0.0453 +/- 0.0073 | 5 |
| RECAP C=1 | Reddit | 0.5950 +/- 0.0020 | 0.0466 +/- 0.0011 | 5 |
| RECAP C=1 | citeseer | 0.9009 +/- 0.0013 | 0.4543 +/- 0.0086 | 5 |
| RECAP C=1 | cora | 0.8228 +/- 0.0032 | 0.4162 +/- 0.0079 | 5 |
| RECAP C=1 | weibo | 0.7533 +/- 0.0064 | 0.3065 +/- 0.0059 | 5 |
| RECAP w/o Adhesion Score | ACM | 0.6849 +/- 0.0189 | 0.1391 +/- 0.0243 | 5 |
| RECAP w/o Adhesion Score | Amazon | 0.5913 +/- 0.0643 | 0.0893 +/- 0.0178 | 5 |
| RECAP w/o Adhesion Score | BlogCatalog | 0.6758 +/- 0.0125 | 0.1918 +/- 0.0273 | 5 |
| RECAP w/o Adhesion Score | Facebook | 0.4984 +/- 0.0336 | 0.0258 +/- 0.0031 | 5 |
| RECAP w/o Adhesion Score | Reddit | 0.5225 +/- 0.0119 | 0.0359 +/- 0.0010 | 5 |
| RECAP w/o Adhesion Score | citeseer | 0.6009 +/- 0.0367 | 0.1039 +/- 0.0305 | 5 |
| RECAP w/o Adhesion Score | cora | 0.5427 +/- 0.0217 | 0.0707 +/- 0.0080 | 5 |
| RECAP w/o Adhesion Score | weibo | 0.8215 +/- 0.0124 | 0.3034 +/- 0.0197 | 5 |
| RECAP w/o Context Score | ACM | 0.7911 +/- 0.0011 | 0.3707 +/- 0.0036 | 5 |
| RECAP w/o Context Score | Amazon | 0.7307 +/- 0.0052 | 0.1322 +/- 0.0030 | 5 |
| RECAP w/o Context Score | BlogCatalog | 0.7306 +/- 0.0061 | 0.3358 +/- 0.0026 | 5 |
| RECAP w/o Context Score | Facebook | 0.5872 +/- 0.0105 | 0.0469 +/- 0.0133 | 5 |
| RECAP w/o Context Score | Reddit | 0.5936 +/- 0.0022 | 0.0470 +/- 0.0006 | 5 |
| RECAP w/o Context Score | citeseer | 0.9010 +/- 0.0029 | 0.4530 +/- 0.0153 | 5 |
| RECAP w/o Context Score | cora | 0.8211 +/- 0.0035 | 0.4040 +/- 0.0177 | 5 |
| RECAP w/o Context Score | weibo | 0.6974 +/- 0.0045 | 0.3025 +/- 0.0085 | 5 |
| RECAP w/o L_H | ACM | 0.7794 +/- 0.0044 | 0.3716 +/- 0.0033 | 5 |
| RECAP w/o L_H | Amazon | 0.7256 +/- 0.0093 | 0.1304 +/- 0.0034 | 5 |
| RECAP w/o L_H | BlogCatalog | 0.7303 +/- 0.0061 | 0.3293 +/- 0.0005 | 5 |
| RECAP w/o L_H | Facebook | 0.5906 +/- 0.0104 | 0.0487 +/- 0.0136 | 5 |
| RECAP w/o L_H | Reddit | 0.5943 +/- 0.0016 | 0.0469 +/- 0.0007 | 5 |
| RECAP w/o L_H | citeseer | 0.9014 +/- 0.0035 | 0.4531 +/- 0.0149 | 5 |
| RECAP w/o L_H | cora | 0.8214 +/- 0.0040 | 0.4063 +/- 0.0171 | 5 |
| RECAP w/o L_H | weibo | 0.8116 +/- 0.0078 | 0.3399 +/- 0.0048 | 5 |
| RECAP w/o L_con | ACM | 0.7907 +/- 0.0015 | 0.3779 +/- 0.0009 | 5 |
| RECAP w/o L_con | Amazon | 0.7178 +/- 0.0231 | 0.1274 +/- 0.0093 | 5 |
| RECAP w/o L_con | BlogCatalog | 0.7366 +/- 0.0011 | 0.3347 +/- 0.0016 | 5 |
| RECAP w/o L_con | Facebook | 0.5776 +/- 0.0081 | 0.0459 +/- 0.0035 | 5 |
| RECAP w/o L_con | Reddit | 0.5950 +/- 0.0039 | 0.0463 +/- 0.0012 | 5 |
| RECAP w/o L_con | citeseer | 0.9017 +/- 0.0013 | 0.4546 +/- 0.0069 | 5 |
| RECAP w/o L_con | cora | 0.8215 +/- 0.0056 | 0.4189 +/- 0.0111 | 5 |
| RECAP w/o L_con | weibo | 0.7952 +/- 0.0042 | 0.3224 +/- 0.0029 | 5 |
| RECAP w/o Residual | ACM | 0.6561 +/- 0.0015 | 0.0994 +/- 0.0015 | 5 |
| RECAP w/o Residual | Amazon | 0.5742 +/- 0.0076 | 0.0837 +/- 0.0020 | 5 |
| RECAP w/o Residual | BlogCatalog | 0.5969 +/- 0.0017 | 0.1000 +/- 0.0018 | 5 |
| RECAP w/o Residual | Facebook | 0.5220 +/- 0.0134 | 0.0230 +/- 0.0006 | 5 |
| RECAP w/o Residual | Reddit | 0.4509 +/- 0.0022 | 0.0292 +/- 0.0002 | 5 |
| RECAP w/o Residual | citeseer | 0.3020 +/- 0.0033 | 0.0401 +/- 0.0026 | 5 |
| RECAP w/o Residual | cora | 0.3292 +/- 0.0040 | 0.0478 +/- 0.0022 | 5 |
| RECAP w/o Residual | weibo | 0.8400 +/- 0.0016 | 0.4249 +/- 0.0041 | 5 |
