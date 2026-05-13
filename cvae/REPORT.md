# Latent Search Implementation Notes

This package is an additive experiment layer. It does not change `instance.py`,
`scheduling/`, `routing/`, or `optimiser/`.

Implemented pieces:

- `cvae.representation`: canonical route-token representation and safe probes
  for VeRoLog symmetries.
- `cvae.data`: loaders for existing valid solution files.
- `cvae.search`: executable continuous latent search. Differential evolution
  explores continuous vectors; a CP-SAT decoder maps each vector to a feasible
  delivery-day schedule; existing routing code evaluates the decoded solution.
  It also has `--routing-fixed-cost`, which temporarily asks OR-Tools to prefer
  fewer active vehicles during route construction while scoring the final file
  with the official instance costs.
- `cvae.model` and `cvae.train`: schedule-level prior/CVAE experiments trained
  on all current instance files with multiple generated target schedules per
  instance.

This is still not the full CVAE-Opt architecture from the paper. The model is a
schedule-level latent model, not an autoregressive route-token decoder. It tests
whether learned latent delivery-day schedules plus exact routing can improve B3
without changing the existing solver core.

Example:

```bash
python -m cvae.search --instance instances/B3.txt --budget 300 --population 24 --iterations 20 --output solutions/cvae_de/B3_solution.txt
```

Initial B3 smoke result after the implementation:

```bash
python -m cvae.search --instance instances/B3.txt --budget 120 --population 4 --iterations 0 --cp-time 1 --route-time 1 --output solutions/cvae_de/B3_smoke_solution.txt
python Validate.py --instance instances/B3.txt --solution solutions/cvae_de/B3_smoke_solution.txt
```

Validated result:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | latent CP-DE smoke | 47 | 226 | 96 102 88 | 2,322,973 | 2,446,774,973 |

Training run:

```bash
python -m cvae.train --instances instances --epochs 120 --batch-size 4 --cp-time 2 --lr 0.0005 --output cvae/checkpoints/schedule_prior_full.pt
```

The first transformer-based prior was unstable on the tiny 20-instance dataset
and produced non-finite losses, so the model was simplified to a request MLP
plus pooled instance context. The stable run trained on all 20 current instance
files and reached:

```text
epoch 120: loss=2.58384 z=2.58015 day=0.03690
```

B3 trained-seed tests:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | trained prior, seed 0 | 47 | 228 | 96 104 89 | 2,472,077 | 2,447,728,077 |
| B3 | trained prior, seed 1 | 47 | 240 | 97 106 93 | 2,589,553 | 2,450,269,553 |

Conclusion for this training pass: the trained schedule prior did not improve on
the untrained latent CP-DE smoke result. The best current valid B3 latent result
at that point was the smoke result.

Expanded CVAE run:

```bash
python -m cvae.train --instances instances --model cvae --samples-per-instance 8 --epochs 80 --batch-size 8 --cp-time 1 --lr 0.0005 --beta 0.001 --output cvae/checkpoints/schedule_cvae_multi.pt
```

This trained on all 20 current instance files and generated 160 target schedules.
The training device was CUDA on the local RTX 4090 Laptop GPU. Final training
line:

```text
epoch 080: loss=2.69541 z=2.67451 day=0.07127
```

B3 with the expanded CVAE checkpoint and high fixed route construction:

```bash
python -m cvae.search --instance instances/B3.txt --checkpoint cvae/checkpoints/schedule_cvae_multi.pt --checkpoint-samples 8 --budget 360 --population 12 --iterations 2 --cp-time 1 --route-time 1 --routing-fixed-cost 100000 --seed 2 --output solutions/cvae_de/B3_cvae_multi_highfixed_solution.txt
python Validate.py --instance instances/B3.txt --solution solutions/cvae_de/B3_cvae_multi_highfixed_solution.txt
```

Validated result:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | CVAE multi + DE + fixed route cost | 47 | 234 | 99 108 93 | 2,508,586 | 2,451,376,586 |

Best valid B3 result after this pass:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | latent CP-DE + fixed route cost | 47 | 216 | 95 101 86 | 2,185,915 | 2,445,217,915 |

Longer CVAE fine-tuning run:

```bash
python -m cvae.train --instances instances --model cvae --resume cvae/checkpoints/schedule_cvae_multi.pt --samples-per-instance 24 --epochs 220 --batch-size 8 --cp-time 1 --lr 0.0002 --beta 0.001 --seed 21 --output cvae/checkpoints/schedule_cvae_multi_long.pt
```

This resumed from the previous CVAE checkpoint, used all 20 available instance
files including B1/B2/B3 plus all `challenge_r100`, `challenge_r200`,
`challenge_r300`, and `challenge_r500` files, and generated 24 target schedules
per instance. Final training line:

```text
epoch 220: loss=2.46333 z=2.43361 day=0.02513
```

B3 search with the longer checkpoint:

```bash
python -m cvae.search --instance instances/B3.txt --checkpoint cvae/checkpoints/schedule_cvae_multi_long.pt --checkpoint-samples 16 --budget 600 --population 16 --iterations 3 --cp-time 1 --route-time 1 --routing-fixed-cost 100000 --seed 22 --output solutions/cvae_de/B3_cvae_multi_long_solution.txt
python Validate.py --instance instances/B3.txt --solution solutions/cvae_de/B3_cvae_multi_long_solution.txt
```

Validated result:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | long CVAE + DE + fixed route cost | 46 | 226 | 104 103 90 | 2,428,922 | 2,401,080,922 |

The canonical file `solutions/cvae_de/B3_solution.txt` now contains this best
valid solution.

Current conclusion: more training across all challenge families did help. It
reduced B3 from the previous best `2,445,217,915` to `2,401,080,922` by finding
a valid 46-vehicle solution. This is still above the user's target area around
2.2B, and the remaining gap is still the max-vehicle term. The next likely
improvement is route-aware training/search that directly learns complete routed
solutions or explicitly optimizes the worst-day route packing.

External elite and archive run:

The user provided a valid B3 solution at repository root. It was copied to
`solutions/external/B3_friend_solution.txt`, added to the persistent archive,
and used as the first B3 training target.

Validated external result:

| Instance | Method | Max vehicles | Vehicle days | Tool use | Distance | Cost |
|---|---:|---:|---:|---|---:|---:|
| B3 | external friend solution | 45 | 235 | 101 104 87 | 2,293,906 | 2,348,763,906 |

Additional friend-aware training:

```bash
python -m cvae.train --instances instances --model cvae --resume cvae/checkpoints/schedule_cvae_multi_long.pt --samples-per-instance 48 --epochs 300 --batch-size 8 --cp-time 1 --lr 0.00012 --beta 0.001 --seed 57 --output cvae/checkpoints/schedule_cvae_friend_long.pt
```

Final line:

```text
epoch 300: loss=2.25617 z=2.21264 day=0.01583
```

This loss is the neural reconstruction objective, not the VeRoLog cost. A
4-worker parallel B3 search with archive enabled saved all incumbent
improvements, but did not beat the external 45-vehicle solution. The canonical
`solutions/cvae_de/B3_solution.txt` now contains the external 45-vehicle file
because it is the best valid B3 solution currently known in this workspace.
