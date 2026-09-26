# Concepts

- **Dataset macro**: average target-dataset metric within each seed, followed
  by mean and population standard deviation over seeds.
- **Domain macro**: average within each target domain per seed, then give each
  domain equal weight before cross-seed aggregation.
- **Frozen score**: a score vector and immutable query mask saved and hashed
  before target labels are available for metric computation.
- **Target context**: target nodes made available to a method before scoring;
  only ARC receives labeled-normal contexts in the locked primary track.
- **OFO supervised evaluation population**: the deterministic stratified 40%
  test partition; train and validation labels are permitted before score
  freeze, while test labels are not.
- **OFO unsupervised evaluation population**: every node in the target graph;
  no labels are available until the full score vector and query mask are
  frozen and hashed.
