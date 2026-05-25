# Contributing

Thanks for looking. This project is open to contributions of pretty much any kind.

## Before you start

This is a student project. I'm not a seasoned maintainer and response times will be what they are. But I'll take every PR seriously and give real feedback.

## How to contribute

**Bug reports:** use the issue template. The more specific the repro, the faster I can look at it.

**Feature ideas:** open an issue first before coding. I don't want anyone wasting time building something I'd push back on. If the issue gets a thumbs up from me, go ahead and PR.

**Pull requests:**

1. Fork the repo
2. Make a branch named something descriptive (`fix/vec-index-collision`, `feat/multilingual-embeddings`)
3. Write or update tests for your change
4. Run the test suite: `python arn_v9/tests/test_all.py`
5. Run the stress test if your change touches memory behavior: `python arn_v9/benchmarks/stress_test.py nano`
6. Open the PR

## Code style

- Python 3.10+
- Keep it readable. If a comment explains *why*, keep it. If it explains *what* the code does, rewrite the code to be clearer.
- I try to match the existing style in each file. If you see inconsistency, that's on me, feel free to fix it.
- No formatter is enforced right now but `black` would be fine if you want to use it.

## Testing

There are two test tiers:

- **Plumbing** — runs without `sentence-transformers`, tests storage, working memory, degraded-mode detection
- **Semantic** — requires the embedding model, tests actual recall quality

The CI runs both. If your change breaks either, fix it before merging.

## Things I'd love help with

See the "Things I think would be valuable next" section in the README. Mem0/Zep comparison benchmarks, async consolidation, and cross-agent sharing are probably the highest value.

## Questions

Just open an issue with the `question` label. Or if you want to reach me privately, my contact info is on my GitHub profile.

## Licensing note for contributors

This project is under **PolyForm Small Business 1.0.0**, not MIT. When you submit a PR, your contribution becomes part of the project under the same license.

What this means in practice:

- You still own the copyright to your contribution
- Your code is licensed to everyone under PolyForm Small Business terms (free for small users, paid for big companies)
- If I ever need to dual-license the project (say, a company wants a commercial license that includes your code), I may need your permission

If any of this is a dealbreaker, open an issue before submitting code and we'll sort it out.

— Mohamed
