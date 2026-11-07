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
