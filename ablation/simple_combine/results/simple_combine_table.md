# Simple Combine Ablation Results

## Average Across Target Graphs

| Method | AUROC | AUPRC | Seeds |
|---|---:|---:|---:|
| RECAP | 0.7447 +/- 0.0024 | 0.2662 +/- 0.0058 | 5 |
| Residual + GMM | 0.6614 +/- 0.0043 | 0.1757 +/- 0.0046 | 5 |
| Residual + KMeans | 0.6782 +/- 0.0034 | 0.1823 +/- 0.0045 | 5 |
| Residual + LOF | 0.5892 +/- 0.0000 | 0.1513 +/- 0.0000 | 5 |
| Residual + Spectral Clustering | 0.5215 +/- 0.0070 | 0.0871 +/- 0.0030 | 5 |

## Per Dataset

| Method | Dataset | AUROC | AUPRC | Seeds |
|---|---|---:|---:|---:|
| RECAP | ACM | 0.7805 +/- 0.0038 | 0.3719 +/- 0.0032 | 5 |
| RECAP | Amazon | 0.7277 +/- 0.0077 | 0.1317 +/- 0.0042 | 5 |
| RECAP | BlogCatalog | 0.7299 +/- 0.0054 | 0.3279 +/- 0.0021 | 5 |
| RECAP | Facebook | 0.5883 +/- 0.0094 | 0.0483 +/- 0.0138 | 5 |
| RECAP | Reddit | 0.5938 +/- 0.0020 | 0.0469 +/- 0.0007 | 5 |
| RECAP | citeseer | 0.9017 +/- 0.0034 | 0.4548 +/- 0.0169 | 5 |
| RECAP | cora | 0.8217 +/- 0.0041 | 0.4056 +/- 0.0169 | 5 |
| RECAP | weibo | 0.8144 +/- 0.0061 | 0.3423 +/- 0.0040 | 5 |
| Residual + GMM | ACM | 0.7286 +/- 0.0023 | 0.1993 +/- 0.0042 | 5 |
| Residual + GMM | Amazon | 0.6410 +/- 0.0038 | 0.0996 +/- 0.0015 | 5 |
| Residual + GMM | BlogCatalog | 0.4995 +/- 0.0058 | 0.1152 +/- 0.0030 | 5 |
| Residual + GMM | Facebook | 0.4902 +/- 0.0100 | 0.0253 +/- 0.0047 | 5 |
| Residual + GMM | Reddit | 0.5598 +/- 0.0070 | 0.0397 +/- 0.0015 | 5 |
| Residual + GMM | citeseer | 0.8409 +/- 0.0188 | 0.3747 +/- 0.0244 | 5 |
| Residual + GMM | cora | 0.7162 +/- 0.0156 | 0.2512 +/- 0.0258 | 5 |
| Residual + GMM | weibo | 0.8150 +/- 0.0029 | 0.3006 +/- 0.0055 | 5 |
| Residual + KMeans | ACM | 0.7628 +/- 0.0015 | 0.2354 +/- 0.0048 | 5 |
| Residual + KMeans | Amazon | 0.7543 +/- 0.0090 | 0.1355 +/- 0.0055 | 5 |
| Residual + KMeans | BlogCatalog | 0.5505 +/- 0.0035 | 0.1315 +/- 0.0031 | 5 |
| Residual + KMeans | Facebook | 0.4686 +/- 0.0227 | 0.0216 +/- 0.0016 | 5 |
| Residual + KMeans | Reddit | 0.5702 +/- 0.0034 | 0.0393 +/- 0.0006 | 5 |
| Residual + KMeans | citeseer | 0.8398 +/- 0.0168 | 0.3692 +/- 0.0287 | 5 |
| Residual + KMeans | cora | 0.7228 +/- 0.0094 | 0.2530 +/- 0.0090 | 5 |
| Residual + KMeans | weibo | 0.7565 +/- 0.0021 | 0.2728 +/- 0.0039 | 5 |
| Residual + LOF | ACM | 0.7655 +/- 0.0000 | 0.3361 +/- 0.0000 | 5 |
| Residual + LOF | Amazon | 0.4431 +/- 0.0000 | 0.0616 +/- 0.0000 | 5 |
| Residual + LOF | BlogCatalog | 0.7040 +/- 0.0000 | 0.3179 +/- 0.0000 | 5 |
| Residual + LOF | Facebook | 0.4468 +/- 0.0000 | 0.0209 +/- 0.0000 | 5 |
| Residual + LOF | Reddit | 0.5518 +/- 0.0000 | 0.0386 +/- 0.0000 | 5 |
| Residual + LOF | citeseer | 0.4739 +/- 0.0000 | 0.0396 +/- 0.0000 | 5 |
| Residual + LOF | cora | 0.7526 +/- 0.0000 | 0.2498 +/- 0.0000 | 5 |
| Residual + LOF | weibo | 0.5759 +/- 0.0000 | 0.1455 +/- 0.0000 | 5 |
| Residual + Spectral Clustering | ACM | 0.4039 +/- 0.0067 | 0.0589 +/- 0.0096 | 5 |
| Residual + Spectral Clustering | Amazon | 0.6178 +/- 0.0301 | 0.0921 +/- 0.0118 | 5 |
| Residual + Spectral Clustering | BlogCatalog | 0.5162 +/- 0.0177 | 0.1788 +/- 0.0206 | 5 |
| Residual + Spectral Clustering | Facebook | 0.4763 +/- 0.0299 | 0.0255 +/- 0.0055 | 5 |
| Residual + Spectral Clustering | Reddit | 0.5237 +/- 0.0039 | 0.0330 +/- 0.0003 | 5 |
| Residual + Spectral Clustering | citeseer | 0.4876 +/- 0.0128 | 0.0459 +/- 0.0026 | 5 |
| Residual + Spectral Clustering | cora | 0.4904 +/- 0.0060 | 0.0557 +/- 0.0014 | 5 |
| Residual + Spectral Clustering | weibo | 0.6558 +/- 0.0248 | 0.2067 +/- 0.0147 | 5 |
