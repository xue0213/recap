# RECAP Interpretability Report: weibo trial 1

## Explanation Metrics

- Total AUPRC: `0.3425`
- Adhesion-only AUPRC: `0.3148`
- Context-only AUPRC: `0.2841`
- Top anomaly lift: `3.8871`
- Context mismatch lift among top nodes: `14.3818`

## Highest-Risk Communities

| Community | Tier | Soft Share | Soft Anomaly Rate | Lift | Mean Score | Top Nodes |
|---:|---|---:|---:|---:|---:|---|
| C21 | high-risk | 0.1901 | 0.2266 | 2.1941 | -0.0123 | 6817;7295;2696;1454;2109 |
| C17 | baseline-risk | 0.0238 | 0.1236 | 1.1973 | 0.0745 | 953;1409;8025;783;515 |
| C7 | baseline-risk | 0.0246 | 0.1159 | 1.1225 | 0.9165 | 2303;3734;4045;1041;5359 |
| C24 | baseline-risk | 0.0223 | 0.0938 | 0.9081 | 0.8413 | 5731;2925;4114;3965;3875 |
| C26 | baseline-risk | 0.0228 | 0.0885 | 0.8567 | 0.6022 | 5588;6826;6056;3536;3941 |
| C14 | baseline-risk | 0.0224 | 0.0868 | 0.8405 | 0.4686 | 5245;1288;4624;4989;7691 |
| C27 | low-risk | 0.0226 | 0.0799 | 0.7741 | 0.0339 | 6598;2373;7646;561;1492 |
| C15 | low-risk | 0.0229 | 0.0768 | 0.7436 | -0.0857 | 7340;4111;4082;3365;3319 |
| C6 | low-risk | 0.0225 | 0.0757 | 0.7328 | nan |  |
| C28 | low-risk | 0.0223 | 0.0757 | 0.7327 | -0.0619 | 990;1381;4756 |

## Top Node Explanations

| Rank | Node | Label | Diagnosis | Community | Adhesion % | Context % | Explanation |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 6817 | 1 | high-risk community prototype outlier | C21 | 1.0000 | 0.9891 | #1 node 6817 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 128.22, at the 100.0% percentile among nodes. Its neighbors only partially support C21 (46.5% neighbor mass; C21:0.465; C32:0.024; C20:0.018), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=98.9%; beta-weighted context contribution=0.094; assignment entropy=0.000. |
| 2 | 7295 | 1 | high-risk community prototype outlier | C21 | 0.9999 | 0.9958 | #2 node 7295 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 118.93, at the 100.0% percentile among nodes. Its neighborhood distribution is different from its own assignment (29.4% neighbor mass on C21; C21:0.294; C1:0.025; C17:0.023), which is a strong context-break signal. Adhesion percentile=100.0%; context percentile=99.6%; beta-weighted context contribution=0.142; assignment entropy=0.000. |
| 3 | 2696 | 1 | neighborhood context break | C21 | 0.9998 | 0.9994 | #3 node 2696 (labeled anomaly) is a neighborhood context break. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 117.25, at the 100.0% percentile among nodes. Its neighborhood distribution is different from its own assignment (14.1% neighbor mass on C21; C21:0.141; C1:0.033; C17:0.032), which is a strong context-break signal. Adhesion percentile=100.0%; context percentile=99.9%; beta-weighted context contribution=0.199; assignment entropy=0.000. |
| 4 | 1454 | 1 | high-risk community prototype outlier | C21 | 0.9996 | 0.9935 | #4 node 1454 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 117.08, at the 100.0% percentile among nodes. Its neighbors only partially support C21 (39.1% neighbor mass; C21:0.391; C32:0.021; C1:0.020), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=99.3%; beta-weighted context contribution=0.113; assignment entropy=0.000. |
| 5 | 2109 | 1 | locally coherent prototype outlier | C21 | 0.9995 | 0.8852 | #5 node 2109 (labeled anomaly) is a locally coherent prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 116.98, at the 100.0% percentile among nodes. Its neighbors mostly agree with C21 (90.2% neighbor mass), so the alert is mainly a within-community residual outlier. Adhesion percentile=100.0%; context percentile=88.5%; beta-weighted context contribution=0.008; assignment entropy=0.000. |
| 6 | 2832 | 1 | high-risk community prototype outlier | C21 | 0.9994 | 0.9706 | #6 node 2832 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 113.49, at the 99.9% percentile among nodes. Its neighbors only partially support C21 (64.5% neighbor mass; C21:0.645; C1:0.015; C32:0.013), adding a moderate context-mismatch signal. Adhesion percentile=99.9%; context percentile=97.1%; beta-weighted context contribution=0.054; assignment entropy=0.000. |
| 7 | 3947 | 1 | neighborhood context break | C21 | 0.9993 | 1.0000 | #7 node 3947 (labeled anomaly) is a neighborhood context break. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 108.48, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (2.6% neighbor mass on C21; C14:0.037; C10:0.037; C17:0.037), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.264; assignment entropy=0.000. |
| 8 | 7575 | 0 | high-risk community prototype outlier | C21 | 0.9992 | 0.9945 | #8 node 7575 (label=0) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 108.11, at the 99.9% percentile among nodes. Its neighbors only partially support C21 (35.9% neighbor mass; C21:0.359; C32:0.019; C13:0.019), adding a moderate context-mismatch signal. Adhesion percentile=99.9%; context percentile=99.5%; beta-weighted context contribution=0.122; assignment entropy=0.000. |
| 9 | 4704 | 1 | neighborhood context break | C21 | 0.9990 | 0.9996 | #9 node 4704 (labeled anomaly) is a neighborhood context break. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 106.07, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (11.4% neighbor mass on C21; C21:0.114; C24:0.027; C34:0.027), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.211; assignment entropy=0.000. |
| 10 | 2753 | 1 | high-risk community prototype outlier | C21 | 0.9989 | 0.9960 | #10 node 2753 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C21 with p=1.000. C21 is high-risk (soft share 19.0%, anomaly rate 22.7%, lift 2.19x). Its distance to the assigned prototype is 103.24, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (29.0% neighbor mass on C21; C21:0.290; C1:0.029; C17:0.023), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=99.6%; beta-weighted context contribution=0.143; assignment entropy=0.000. |

## Figures

- `../figures/weibo_trial1_paper_confident_communities_tsne.pdf`
- `../figures/weibo_trial1_paper_high_risk_response_maps.pdf`
- `../figures/weibo_trial1_paper_community_mass_lift.pdf`
- `../figures/weibo_trial1_paper_community_lift_bars.pdf`
- `../figures/weibo_trial1_pretrain_residual_embedding.pdf`
- `../figures/weibo_trial1_trained_residual_communities.pdf`
- `../figures/weibo_trial1_community_risk_map.pdf`
- `../figures/weibo_trial1_anomaly_score_map.pdf`
- `../figures/weibo_trial1_score_component_percentiles.pdf`
- `../figures/weibo_trial1_community_lift_bars.pdf`
