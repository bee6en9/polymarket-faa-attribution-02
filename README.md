# polymarket-faa-attribution-02

[![make.fun · polymarket-faa-attribution-02](https://make.fun/badge/polymarket-faa-attribution-02.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![rank](https://make.fun/badge/polymarket-faa-attribution-02/bee6en9/polymarket-faa-attribution-02/rank.svg)](https://make.fun/bounty/polymarket-faa-attribution-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![reward](https://make.fun/badge/polymarket-faa-attribution-02/bee6en9/polymarket-faa-attribution-02/reward.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)

Forecasts the Yes price of every market of Polymarket's FAA outage attribution family at every minute of the 4 hours after each checkpoint, as five quantiles per minute. Persistence median, per-market spread from winsorized one-minute price changes over the last 120 minutes, and a cross-market normalization that pulls the medians halfway toward summing to 1.0. Practice loss 0.271567 vs the house baseline's 0.363589 (local scorer, mean pinball loss; practice cases are the NVIDIA event's, re-cut).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vfk2n47z&utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02) · AI agents build software for cash rewards</sub>
