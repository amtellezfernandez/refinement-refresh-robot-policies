# Protocol E evidence binding

A PASSED evaluator item must reference the retained record of the counted native
Protocol E check for its own model. Existing native entry points already separate
these scopes:

| Model | Required authorization scope | Record `stack` |
| --- | --- | --- |
| pi05 | PROTOCOL_E_PI05 | pi05 |
| smolvla | PROTOCOL_E_SMOLVLA | smolvla |

The authorization's existing approved_by and file hash checks remain. The
qualification reader also checks the scope, the record's stack, and that
`bindings.protocol_e_record_sha256` equals the hash of the evidence file actually
read. Its verdict must be EVALUATOR_QUALIFIED, as before. These checks reconcile
existing fields; no extra approval, native run or evidence schema is introduced.

PROTOCOL_E0_PI05 is a non-counting diagnostic scope. Even a file that claims a
qualified verdict cannot use that authorization to qualify either evaluator.
Evidence from the other stack likewise cannot qualify this one. Recomputed
file hashes do not repair a scope/stack mismatch.

These consistency checks do not prove that the recorded execution occurred or
that its evaluator is correct. Trusted authorization artifacts, independent
Protocol E checks, complete fixture coverage and retained native execution
records are still required. Other qualification items retain their existing
validation. No shipped UNVERIFIED record is changed by this correction; native
qualification and campaign release remain outstanding.

Offline regression fixtures are fabricated in temporary directories. They are
not E0 artifacts, permissions, qualification evidence or confirmation identities.
