# RECAP Interpretability Report: weibo trial 2

## Explanation Metrics

- Total AUPRC: `0.3378`
- Adhesion-only AUPRC: `0.2904`
- Context-only AUPRC: `0.3137`
- Top anomaly lift: `3.3121`
- Context mismatch lift among top nodes: `15.6069`

## Highest-Risk Communities

| Community | Tier | Soft Share | Soft Anomaly Rate | Lift | Mean Score | Top Nodes |
|---:|---|---:|---:|---:|---:|---|
| C33 | elevated-risk | 0.2498 | 0.1868 | 1.8087 | -0.0349 | 3947;2109;3706;7575;1454 |
| C4 | elevated-risk | 0.0286 | 0.1700 | 1.6457 | 0.3290 | 2303;5245;5731;3734;2925 |
| C2 | baseline-risk | 0.0218 | 0.0985 | 0.9543 | 0.0429 | 2425;7748;515;4773;7520 |
| C17 | low-risk | 0.0210 | 0.0802 | 0.7769 | 2.7797 | 2696;7286;684;6038;7735 |
| C30 | low-risk | 0.0208 | 0.0791 | 0.7658 | 0.8656 | 5212;2357;7114;5965;1574 |
| C26 | low-risk | 0.0204 | 0.0789 | 0.7636 | -0.0902 | 1072;5375;337;2971;4718 |
| C35 | low-risk | 0.0207 | 0.0781 | 0.7567 | 3.6672 | 1097;2319 |
| C27 | low-risk | 0.0205 | 0.0771 | 0.7469 | -0.1039 | 3038;4202;3176;3984;4111 |
| C29 | low-risk | 0.0205 | 0.0767 | 0.7430 | -0.1022 | 7521;3853;907;332;6053 |
| C11 | low-risk | 0.0206 | 0.0767 | 0.7425 | 4.8076 | 7979;5664;965;7354;3886 |

## Top Node Explanations

| Rank | Node | Label | Diagnosis | Community | Adhesion % | Context % | Explanation |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 3947 | 1 | locally coherent prototype outlier | C33 | 1.0000 | 0.9480 | #1 node 3947 (labeled anomaly) is a locally coherent prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 144.34, at the 100.0% percentile among nodes. Its neighbors mostly agree with C33 (80.3% neighbor mass), so the alert is mainly a within-community residual outlier. Adhesion percentile=100.0%; context percentile=94.8%; beta-weighted context contribution=0.026; assignment entropy=0.000. |
| 2 | 2109 | 1 | locally coherent prototype outlier | C33 | 0.9999 | 0.8435 | #2 node 2109 (labeled anomaly) is a locally coherent prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 124.39, at the 100.0% percentile among nodes. Its neighbors mostly agree with C33 (95.0% neighbor mass), so the alert is mainly a within-community residual outlier. Adhesion percentile=100.0%; context percentile=84.4%; beta-weighted context contribution=0.002; assignment entropy=0.000. |
| 3 | 3706 | 1 | high-risk community prototype outlier | C33 | 0.9998 | 0.9901 | #3 node 3706 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 116.08, at the 100.0% percentile among nodes. Its neighbors only partially support C33 (43.8% neighbor mass; C33:0.438; C35:0.047; C25:0.021), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=99.0%; beta-weighted context contribution=0.105; assignment entropy=0.000. |
| 4 | 7575 | 0 | high-risk community prototype outlier | C33 | 0.9995 | 0.9955 | #4 node 7575 (label=0) is a high-risk community prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 111.17, at the 100.0% percentile among nodes. Its neighborhood distribution is different from its own assignment (30.5% neighbor mass on C33; C33:0.305; C10:0.025; C22:0.024), which is a strong context-break signal. Adhesion percentile=100.0%; context percentile=99.5%; beta-weighted context contribution=0.144; assignment entropy=0.000. |
| 5 | 1454 | 1 | high-risk community prototype outlier | C33 | 0.9996 | 0.9931 | #5 node 1454 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 111.18, at the 100.0% percentile among nodes. Its neighbors only partially support C33 (38.3% neighbor mass; C33:0.383; C9:0.046; C2:0.036), adding a moderate context-mismatch signal. Adhesion percentile=100.0%; context percentile=99.3%; beta-weighted context contribution=0.120; assignment entropy=0.000. |
| 6 | 2696 | 1 | neighborhood context break | C17 | 0.9993 | 0.9996 | #6 node 2696 (labeled anomaly) is a neighborhood context break. It assigns to C17 with p=0.979. C17 is low-risk (soft share 2.1%, anomaly rate 8.0%, lift 0.78x). Its distance to the assigned prototype is 109.22, at the 99.9% percentile among nodes. Its neighborhood distribution is different from its own assignment (3.8% neighbor mass on C17; C33:0.065; C17:0.038; C0:0.035), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.251; assignment entropy=0.039. |
| 7 | 4704 | 1 | prototype-assignment tension | C12 | 0.9994 | 0.9985 | #7 node 4704 (labeled anomaly) is a prototype-assignment tension. It assigns to C12 with p=0.387. C12 is low-risk (soft share 2.0%, anomaly rate 7.4%, lift 0.72x). The closest prototype is C4 (distance 108.56), but the soft assignment favors C12; this assignment/prototype tension makes the case more diagnostic. Its neighborhood distribution is different from its own assignment (2.9% neighbor mass on C12; C33:0.059; C0:0.030; C7:0.030), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=99.8%; beta-weighted context contribution=0.189; assignment entropy=0.459. |
| 8 | 7295 | 1 | prototype-assignment tension | C25 | 0.9992 | 1.0000 | #8 node 7295 (labeled anomaly) is a prototype-assignment tension. It assigns to C25 with p=0.949. C25 is low-risk (soft share 2.7%, anomaly rate 6.1%, lift 0.59x). The closest prototype is C4 (distance 104.81), but the soft assignment favors C25; this assignment/prototype tension makes the case more diagnostic. Its neighborhood distribution is different from its own assignment (2.2% neighbor mass on C25; C33:0.068; C22:0.033; C7:0.032), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=100.0%; beta-weighted context contribution=0.259; assignment entropy=0.070. |
| 9 | 6817 | 1 | high-risk community prototype outlier | C33 | 0.9990 | 0.9910 | #9 node 6817 (labeled anomaly) is a high-risk community prototype outlier. It assigns to C33 with p=1.000. C33 is elevated-risk (soft share 25.0%, anomaly rate 18.7%, lift 1.81x). Its distance to the assigned prototype is 105.61, at the 99.9% percentile among nodes. Its neighbors only partially support C33 (41.9% neighbor mass; C33:0.419; C25:0.026; C5:0.020), adding a moderate context-mismatch signal. Adhesion percentile=99.9%; context percentile=99.1%; beta-weighted context contribution=0.111; assignment entropy=0.000. |
| 10 | 5212 | 0 | prototype-assignment tension | C30 | 0.9989 | 0.9992 | #10 node 5212 (label=0) is a prototype-assignment tension. It assigns to C30 with p=0.618. C30 is low-risk (soft share 2.1%, anomaly rate 7.9%, lift 0.77x). The closest prototype is C4 (distance 103.27), but the soft assignment favors C30; this assignment/prototype tension makes the case more diagnostic. Its neighborhood distribution is different from its own assignment (2.2% neighbor mass on C30; C33:0.240; C35:0.039; C12:0.039), which is a strong context-break signal. Adhesion percentile=99.9%; context percentile=99.9%; beta-weighted context contribution=0.233; assignment entropy=0.237. |

## Figures

- `../figures/weibo_trial2_paper_confident_communities_tsne.pdf`
- `../figures/weibo_trial2_paper_high_risk_response_maps.pdf`
- `../figures/weibo_trial2_paper_community_mass_lift.pdf`
- `../figures/weibo_trial2_paper_community_lift_bars.pdf`
- `../figures/weibo_trial2_pretrain_residual_embedding.pdf`
- `../figures/weibo_trial2_trained_residual_communities.pdf`
- `../figures/weibo_trial2_community_risk_map.pdf`
- `../figures/weibo_trial2_anomaly_score_map.pdf`
- `../figures/weibo_trial2_score_component_percentiles.pdf`
- `../figures/weibo_trial2_community_lift_bars.pdf`
