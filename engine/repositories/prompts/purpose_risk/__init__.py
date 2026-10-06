"""The purpose/risk prompt: `prompt.md`, the rules in `checks.py`, and how it ends.

The prompt asks for `P: claim` lines closed by `END`, and everything that
format decides lives in this folder, so a change to the format changes one
place.

`END` is a stop sequence, not decoration: free-form lines have no natural end,
and a model that never emits `<|im_end|>` runs to `MAX_TOKENS`. The server
strips the stop sequence, so a finished answer arrives without it.
"""

STOP = ("<|im_end|>", "\nEND")

# Twice the longest healthy answer seen over gold1000-r1 (p99 333, one label cut
# off at 400); a runaway holds a slot ~40 s. The cap is taken out of `slot_ctx`,
# so prompts up to ~4,250 tokens still fit. Labels cut off here go to a second
# run with a larger `slot_ctx` and cap.
MAX_TOKENS = 800
