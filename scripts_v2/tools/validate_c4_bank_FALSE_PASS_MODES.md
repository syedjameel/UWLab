# How validate_c4_bank.py can be handed a bad bank and say PASS

Not how the tool works — how it can be wrong. Written for whoever reviews an Arm 1/2/3 bank's
verdict, since three arms are about to be judged on this instrument. If a verdict looks
suspiciously clean, check these first before trusting it.

1. **Wrong pair / wrong mating-frame constants, and nothing catches it.** `_LEG_OFF_POS`,
   `_RECV_OFF_QUAT_WXYZ`, etc. are hardcoded for exactly one pair (SquareTableLeg200mmDecomp /
   OneLegInsertionFixture). The Part-A self-check only proves the reprojection MATH is internally
   consistent (a synthetic assembled pose scores 0.00e+00) — it says nothing about whether those
   constants are the RIGHT constants for the bank under test. A bank generated for a different
   leg/fixture pair, or against a since-revised `assembled_offset` in that asset's metadata.yaml,
   reprojects through silently-wrong numbers and can land "in band" by coincidence. Same risk on
   the staging side: `_stage_bank_into_scratch_layout` derives `pair` from the ENV's resolved USD
   paths, not from the bank file's own contents, so a bank authored for one pair but tested with
   `--bank-path` pointed at it while the env's `--gain-regime` cfg spawns a DIFFERENT pair would
   copy and load without any error, using the wrong leg/fixture geometry underneath it.

2. **n too small for the held-fraction PASS/FAIL to mean anything.** No confidence interval is
   computed anywhere in this tool — `held_fraction_contact_graded` is a bare point estimate, and
   the gate is a hard `< 0.70`. At `n_envs=128`, a true rate of 0.65–0.75 can land on either side of
   the gate from sampling noise alone (Wilson 95% half-width ~8 points at that n). `n_envs` states
   are also drawn WITH REPLACEMENT from the whole bank (MultiResetManager's own convention), so
   small `n_envs` effectively sees fewer than `n_envs` distinct states, tightening this further. A
   single default `--seed` compounds it — nothing here tells you whether a different sample of the
   same bank would flip the verdict.

3. **Grip held by something other than the fingers.** `held(contact)` only asks "are the five
   DELTO tips loaded above 0.2N, opposed" — it never asks whether the fingers are the DOMINANT
   support. A leg wedged tight enough in the bore to be held by fixture/bore friction alone can
   still read positive on this check if a finger is merely resting against it above threshold while
   contributing little of the actual holding force. This is the same failure from two directions:
   `depth_delta` can look great (bore friction, not the hand, is keeping it seated) while
   `held(contact)` also reads high (incidental finger contact), and neither number on its own
   reveals that the hand isn't actually doing the work.

4. **A high death rate hides under a passing held-fraction.** Envs that terminate
   (`abnormal_robot`) get their running buffer FROZEN at the last-alive reading, by design — so a
   bank that causes MANY explosive/abnormal-robot failures can still show a fine
   `held(contact)`/`depth_delta_mm` at 2s, because those dead envs contribute whatever they read a
   split second before catastrophic failure, not a penalty for having failed. `died_fraction` and
   `died_reason_counts` ARE printed and stored, but nothing in the PASS/FAIL gate reads them — a
   reviewer who only checks the two headline numbers will miss this.

5. **Part A and Part B sample different states, silently.** Part A draws `n_states_a` WITHOUT
   replacement (numpy RNG); Part B draws `n_envs` WITH replacement (MultiResetManager's own torch
   RNG, a separate stream). For a bank with real intra-bank heterogeneity — e.g. any of the
   `_MERGED` multi-generation-run banks seen on this box — Part A's depth read and Part B's hold
   read are not guaranteed to describe the same slice of the bank. A combined "depth in band AND
   holds" verdict can be true of the BANK IN AGGREGATE while false of any single state.

6. **Isolated self-check PASSING is not evidence about the bank.** The UWLab-xp05.9 fix (reading
   the action term's `.processed_actions` pre-physics) verifies the HARNESS's action pathway, not
   the bank's grasp quality — it PASSES on garbage banks too (confirmed: it passes on the known-bad
   IK bank). Don't read a green isolated-check line as supporting evidence for a PASS verdict; it
   only rules out one specific way the verdict could be wrong for the wrong reason.

7. **Gain-regime / contact-threshold mismatch.** `--gain-regime train` and
   `--contact-threshold-n 0.2` are both inherited defaults, not derived per-arm. Testing a bank
   under the wrong OSC gain regime, or against a threshold tuned for a different gripper mass/
   geometry than an arm actually uses, can shift the held-fraction number in either direction for
   reasons that have nothing to do with the bank's grasp quality.

8. **Schema validation is shape-only, not sanity.** `validate_bank_schema` checks key names and
   equal field lengths — it does not check for NaN/Inf poses, degenerate (unnormalized/zero)
   quaternions, or a `joint_position_target` that silently fell back to some default posture. A
   corrupted-but-well-shaped bank can pass schema validation and still produce a plausible-looking
   number by coincidence.

9. **A stray `--self-check-action zero` run.** That flag exists only for the UWLab-xp05.9 A/B
   diagnostic. Left set to `zero` on a real evaluation invocation, it burns one limp-handed step
   before the real rollout starts, biasing the whole 2s hold measurement worse than the bank
   actually is — the mirror-image risk (false FAIL, not false PASS), but worth checking if a
   verdict looks surprisingly bad too.

None of these are hypothetical dangers invented for this note — #2, #3/#4, and #6 are all direct
consequences of design decisions made while building this tool that were correct trade-offs for
THIS validation task, not universal safety properties of the numbers it prints.
