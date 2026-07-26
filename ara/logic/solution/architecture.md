# Architecture

The Phase 2 harness separates:

1. declarative source/target/run manifests;
2. method-native model adapters;
3. audited source and target label access;
4. atomic score/checkpoint artifacts;
5. independent artifact validation and aggregation.

The final analyzer reads only completed formal run directories. Failed and
superseded attempts remain outside that directory and cannot enter aggregates.
