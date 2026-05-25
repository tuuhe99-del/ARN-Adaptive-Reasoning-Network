# ARN Task: test and fix

## Task ID
`ARN-test-and-fix`

## Review Chain
```
codex → kimi → claude
```

## Description

digest full arn code and how it works then run test on red team openclaw agent run different sessions and every session talk about something different including identity, yourself aka user, the system aswell as let it run tool calls everytime a new session test if it can remember information from and old sessions fix all issues that come up to get arn to full operational memory system arn should be able to remember tool calls procedures identities how information relates, what information relates to its task etc  once your done pass what you implemented to the next agent in the turn by the proper channels and tell them to run the tests and fixes aswell

## Agent Instructions

- Read `COLLAB.md` and `docs/collab-protocol.md` first.
- Claim your step, do minimal correct work, write a handoff.
- Run `python3 -m py_compile` on every Python file you change.

## Verification
```bash
python3 -m pytest arn_v9/tests/ -x -q 2>&1 | tail -10
```
