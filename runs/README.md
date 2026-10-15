# runs/

Training output lives here. Everything in this directory is generated — you can
delete it and regenerate it at any time.

Ignored by git except for the shipped reference policy:

```
runs/champion/champion.json       the reference policy (loaded by the browser's
                                  "Load champion" button and by scripts/play.mjs)
runs/champion/champion.meta.json  how it was chosen and how it scored
runs/champion/*.png               figures for it
```

## Regenerating

```bash
node scripts/train.mjs --lanes 2 --steps 800000 --out runs/main   # train
node scripts/champion.mjs --dir runs/main --force                # rank + pick
cp runs/main/champion.json runs/champion/champion.json           # ship it
```

## What train.mjs writes

| File | Contents |
| --- | --- |
| `laneN-best.json` | best-scoring checkpoint per lane (greedy evaluation) |
| `laneN-latest.json` | most recent checkpoint |
| `laneN-final.json` | checkpoint at the end of the run |
| `laneN-history.json` | `[steps, distance]` per episode, for plotting |
| `laneN-best.meta.json` | when and how that checkpoint was produced |
| `run.json` | the whole session: args plus every lane's summary and curve |
