"""The chapter's own library, kept apart from the scripts that drive it.

`splatting.py` (Listing 9.1) lives here; the drivers next door import it as
`from deps.splatting import ...`, which works from any directory because Python
puts the running script's folder on `sys.path`.
"""
