# Related Work

The Phase 2 comparison covers the official implementations of ARC, UNPrompt,
AnomalyGFM, and IA-GGAD. Exact pinned revisions and archive hashes are recorded
in `rebuttal/baselines/upstream_manifest.json`.

The OFO comparison covers GADBench's supervised GCN/GAT/BWGNN/XGBGraph
recipes and the official DOMINANT, AnomalyDAE, CoLA, ADA-GAD, BWGNN, and
PyGOD releases. Pinned revisions and archive hashes are recorded in
`rebuttal/ofo_baselines/upstream_manifest.json`.

The final extension covers the official DiffGAD, GUIDE, and OWLEYE releases,
plus ORCA for exact GUIDE orbit counts. Pinned revisions and archive hashes
are recorded in `rebuttal/new_baselines/upstream_manifest.json`. Because the
three model repositories publish no license file, the artifact contains
independently written compatibility adapters rather than redistributed source.
