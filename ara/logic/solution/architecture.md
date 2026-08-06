# Architecture

The Phase 2 harness separates:

1. declarative source/target/run manifests;
2. method-native model adapters;
3. audited source and target label access;
4. atomic score/checkpoint artifacts;
5. independent artifact validation and aggregation.

The final analyzer reads only completed formal run directories. Failed and
superseded attempts remain outside that directory and cannot enter aggregates.

The 12-dataset OFO harness adds a declarative 288-run Cartesian manifest,
method-specific supervised/unsupervised label vaults, deterministic split and
large-graph adapters, one atomic result directory per run, and an analyzer
that independently requires exactly the complete manifest before aggregation.

The three-baseline extension uses a separate immutable 81-run manifest,
upstream and raw-data hash gates, method-specific supervision vaults, cached
exact ORCA motifs and official OWLEYE features, atomic frozen-score artifacts,
and an independent 126-evaluation analyzer. The protocol-wide consolidator
joins only accepted analysis records and preserves their evaluation strata.
