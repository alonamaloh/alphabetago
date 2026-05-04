# alphabetago

Research project: alpha-beta search for Go, guided by a neural net with information-bit deepening (path cost = sum of `-log2(policy_prob)` along the path).

Network: policy head + ownership head; expected score is the sum of predicted point ownerships. Tromp-Taylor-style area scoring. Zero-style self-play, starting on 9×9.

Status: scaffold only — nothing implemented yet.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
