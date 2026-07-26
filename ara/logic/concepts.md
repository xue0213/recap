# Concepts

- **Dataset macro**: average target-dataset metric within each seed, followed
  by mean and population standard deviation over seeds.
- **Domain macro**: average within each target domain per seed, then give each
  domain equal weight before cross-seed aggregation.
- **Frozen score**: a score vector and immutable query mask saved and hashed
  before target labels are available for metric computation.
- **Target context**: target nodes made available to a method before scoring;
  only ARC receives labeled-normal contexts in the locked primary track.
