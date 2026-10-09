so-vits-svc
===========

Original docs are in [README\_ORIGINAL.md](./README_ORIGINAL.md).

This is a snapshot of `f22819cb8718ead5e50dea4fdbd05195c72fb1a8` (branch `origin/4.0`). It was taken
from justinjohn0306's fork at https://github.com/justinjohn0306/so-vits-svc


Dev Installation
----------------

Use our `requirements.txt` instead of the original `upstream-requirements-but-broken.txt`

Installing this may give you trouble. Brandon had trouble under Python3.10, so downgraded to Python 3.9
using the deadsnakes PPA. Furthermore, the version of pyworld originally specified refused to build;
upgrading to the latest point release fixed it.

`pyworld==0.2.5` -> `pyworld==0.2.12`

tl;dr: 

- Python3.9
- pyworld 0.2.12


