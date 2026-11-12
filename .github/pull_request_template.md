**What and why**

**Tests**
- [ ] `python -m pytest -q tests`
- [ ] `python -m pyflakes *.py tests/*.py scripts/*.py`
- [ ] `python chaos_test.py --mock --phases 1,2,3` (if this touches leases, preemption, recovery or the worker loop)

**Numbers** (if any): mock or live? model? hardware?
