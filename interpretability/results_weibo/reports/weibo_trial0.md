# RECAP Interpretability Report: weibo trial 0

## Explanation Metrics

- Total AUPRC: `0.3479`
- Adhesion-only AUPRC: `0.3065`
- Context-only AUPRC: `0.3381`
- Top anomaly lift: `3.7031`
- Context mismatch lift among top nodes: `21.9089`

## Highest-Risk Communities

| Community | Tier | Soft Share | Soft Anomaly Rate | Lift | Mean Score | Top Nodes |
|---:|---|---:|---:|---:|---:|---|
| C8 | high-risk | 0.1525 | 0.2366 | 2.2912 | -0.0180 | 2109;1454;3947;3706;6817 |
| C32 | high-risk | 0.0261 | 0.2158 | 2.0892 | 0.3970 | 5245;2303;5731;3734;2925 |
| C0 | baseline-risk | 0.0256 | 0.0963 | 0.9323 | 0.0230 | 8025;2230;783;5362;515 |
| C20 | baseline-risk | 0.0306 | 0.0953 | 0.9224 | -0.0538 | 3230;7319;8093;5041;4103 |
| C27 | baseline-risk | 0.0230 | 0.0896 | 0.8674 | 1.5550 | 6598;965;7354;3886;7589 |
| C29 | baseline-risk | 0.0245 | 0.0894 | 0.8658 | 2.3526 | 480;1073;1097;7145;2373 |
| C14 | baseline-risk | 0.0234 | 0.0832 | 0.8054 | -0.0743 | 7646;1978;4688;6010;4962 |
| C22 | low-risk | 0.0236 | 0.0796 | 0.7708 | -0.0027 | 1022;7076;8302;3078;6126 |
| C11 | low-risk | 0.0232 | 0.0770 | 0.7460 | -0.0953 | 7202;3619;3433;5965;1766 |
| C9 | low-risk | 0.0227 | 0.0764 | 0.7396 | nan |  |

## Top Node Explanations

| Rank | Node | Label | Diagnosis | Community | Adhesion % | Context % | Explanation |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 2109 | 1 | locally coherent prototype outlier | C8 | 1.0000 | 0.8679 | #1 node 2109 (labeled anomaly) is a locally coherent prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 144.03, at the 100.0% percentile among nodes. Its neighbors mostly agree with C8 (94.3% neighbor mass), so the alert is mainly a within-community residual outlier. Adhesion percentile=100.0%; context percentile=86.8%; beta-weighted context contribution=0.002; assignment entropy=0.000. |
| 2 | 1454 | 1 | high-risk community prototype outlier | C8 | 0.9999 | 0.9880 | #2 node 1454 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 134.06, at the 100.0% percentile among nodes. Its neighbors only partially support C8 (41.4% neighbor mass; C8:0.414; C29:0.027; C20:0.022), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=98.8%; beta-weighted context contribution=0.096; assignment entropy=0.000. |
| 3 | 3947 | 1 | locally coherent prototype outlier | C8 | 0.9998 | 0.9543 | #3 node 3947 (labeled anomaly) is a locally coherent prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 132.30, at the 100.0% percentile among nodes. Its neighbors mostly agree with C8 (74.6% neighbor mass), so the alert is mainly a within-community residual outlier. Adhesion percentile=100.0%; context percentile=95.4%; beta-weighted context contribution=0.031; assignment entropy=0.000. |
| 4 | 3706 | 1 | high-risk community prototype outlier | C8 | 0.9996 | 0.9896 | #4 node 3706 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 126.76, at the 100.0% percentile among nodes. Its neighbors only partially support C8 (36.4% neighbor mass; C8:0.364; C29:0.051; C20:0.023), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=99.0%; beta-weighted context contribution=0.108; assignment entropy=0.000. |
| 5 | 6817 | 1 | high-risk community prototype outlier | C8 | 0.9995 | 0.9874 | #5 node 6817 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 126.06, at the 100.0% percentile among nodes. Its neighbors only partially support C8 (42.2% neighbor mass; C8:0.422; C18:0.025; C5:0.021), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=98.7%; beta-weighted context contribution=0.094; assignment entropy=0.000. |
| 6 | 7295 | 1 | prototype-assignment tension | C31 | 0.9994 | 0.9988 | #6 node 7295 (labeled anomaly) is a prototype-assignment tension. It assigns to C31 with p=0.479. C31 is low-risk (soft share 2.4%, anomaly rate 6.9%, lift 0.67x). The closest prototype is C29 (distance 112.77), but the soft assignment favors C31; this assignment/prototype tension makes the case more diagnostic. Its neighborhood distribution is different from its own assignment (2.3% neighbor mass on C31; C8:0.093; C1:0.074; C29:0.039), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=99.9%; beta-weighted context contribution=0.190; assignment entropy=0.328. |
| 7 | 5168 | 1 | high-risk community prototype outlier | C8 | 0.9993 | 0.9723 | #7 node 5168 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 111.31, at the 99.9% percentile among nodes. Its neighbors only partially support C8 (63.6% neighbor mass; C8:0.636; C27:0.036; C29:0.030), adding a moderate context-mismatch signal. Adhesion percentile=99.9%; context percentile=97.2%; beta-weighted context contribution=0.050; assignment entropy=0.000. |
| 8 | 7575 | 0 | high-risk community prototype outlier | C8 | 0.9992 | 0.9925 | #8 node 7575 (label=0) is a high-risk community prototype outlier. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 110.26, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (30.0% neighbor mass on C8; C8:0.300; C1:0.040; C0:0.038), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=99.3%; beta-weighted context contribution=0.125; assignment entropy=0.000. |
| 9 | 2696 | 1 | neighborhood context break | C8 | 0.9990 | 0.9999 | #9 node 2696 (labeled anomaly) is a neighborhood context break. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 109.74, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (1.3% neighbor mass on C8; C24:0.082; C9:0.078; C11:0.072), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.247; assignment entropy=0.000. |
| 10 | 4704 | 1 | neighborhood context break | C8 | 0.9989 | 0.9995 | #10 node 4704 (labeled anomaly) is a neighborhood context break. It assigns to C8 with p=1.000. C8 is high-risk (soft share 15.3%, anomaly rate 23.7%, lift 2.29x). Its distance to the assigned prototype is 108.45, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (7.0% neighbor mass on C8; C8:0.070; C27:0.040; C2:0.030), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.211; assignment entropy=0.000. |

## Figures

- `../figures/weibo_trial0_paper_confident_communities_tsne.pdf`
- `../figures/weibo_trial0_paper_high_risk_response_maps.pdf`
- `../figures/weibo_trial0_paper_community_mass_lift.pdf`
- `../figures/weibo_trial0_paper_community_lift_bars.pdf`
- `../figures/weibo_trial0_pretrain_residual_embedding.pdf`
- `../figures/weibo_trial0_trained_residual_communities.pdf`
- `../figures/weibo_trial0_community_risk_map.pdf`
- `../figures/weibo_trial0_anomaly_score_map.pdf`
- `../figures/weibo_trial0_score_component_percentiles.pdf`
- `../figures/weibo_trial0_community_lift_bars.pdf`
