# How problem difficulty was calibrated

The brief warns that easy problems teach a bandit nothing, because strong models solve all of them.
A three-problem smoke test confirmed it: at `reasoning.effort=low`, the original hardest tier was
solved by nearly every model. The tiers were rebuilt, and then calibrated on separate pilot problems.

## Rule, written before the first pilot ran

- **Target:** mean strict accuracy over the three models lands between 40% and 85% in every family,
  and tier 3 is harder than tier 1 in every family.
- **May change:** templates, parameter ranges, and tier counts, for a whole family at a time.
- **May not change in response to a pilot:** the models, the prompt, the verifier, or any choice
  made because of which model won or how far apart the models were. Specialization is what is being
  measured, so it must not be tuned.
- **Isolation:** pilot problems use a different seed from the main set and are not scored in any
  result. The main set is generated and run once, after calibration ends.

Pilot responses stay in `data/pilot*/` so this can be audited. The outcome of each pilot is below.

## Outcome

- **Smoke test** (`data/smoke_v1`, `data/smoke`): `reasoning: disabled` was rejected by most frontier
  endpoints, so the shared setting became `reasoning.effort=low`. At that setting, the original tier 3
  was solved by nearly every model.
- **Pilot 1** (`data/pilot1`, seed 7, with the tiers at the time): every family was above or near the
  ceiling of the target range. Integrals missed it.
- **Pilot 2** (`data/pilot2`, seed 8, with the final tiers): every family was within 40-85%, and tier 3
  was below tier 1 in every family. Calibration stopped here, and the main set (seed 20261107) was run
  once.

`make reproduce` rescores both pilots into `results/pilots.json`, so the exact figures can be checked.
The models, prompt and verifier were the same in both pilots and the main run.
