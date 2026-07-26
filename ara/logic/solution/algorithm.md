# Algorithm

For each locked method, setting, and seed:

1. load source graphs and method-native aligned features;
2. train one shared checkpoint across the setting's sources;
3. select any permitted fusion weight on seed-0 source scores only;
4. freeze the selected weight across targets and remaining seeds;
5. produce and hash each target score and query mask;
6. unlock target labels only for AUROC/AUPRC computation;
7. reload the checkpoint and compare reproduced scores;
8. independently recompute every final metric during aggregation.
