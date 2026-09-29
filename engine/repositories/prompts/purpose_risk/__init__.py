"""The purpose/risk prompt: `prompt.md`, the rules in `checks.py`, and how it ends.

The prompt asks for `P: claim | 1-5` lines closed by `END`, and everything that
format decides lives in this folder, so a change to the format changes one
place.

`END` is a stop sequence, not decoration: free-form lines have no natural end,
and a model that never emits `<|im_end|>` runs to `MAX_TOKENS`. The server
strips the stop sequence, so a finished answer arrives without it.
"""

STOP = ("<|im_end|>", "\nEND")

# Enough for the longest healthy answer seen (186 tokens over ten labels) with
# room to spare; low enough that a runaway costs seconds, not a minute.
MAX_TOKENS = 400
