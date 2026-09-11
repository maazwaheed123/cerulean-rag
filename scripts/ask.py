"""Ask one question:  python scripts/ask.py "question" [--json] [--show-context] [--model TAG]"""

import sys

from cerulean_rag.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    # global options (--model, --log-level) must precede the sub-command for argparse
    globals_ = []
    rest = []
    it = iter(argv)
    for a in it:
        if a in ("--model", "--log-level"):
            globals_ += [a, next(it, "")]
        else:
            rest.append(a)
    sys.exit(main(globals_ + ["ask"] + rest))
