#!/usr/bin/python3
"""Bootstrap the book's third-party dependencies into `deps/`.

It lives in `Chapter09/` because this is the only chapter with a C++ build, but the tree it populates is one level up. Both paths below are
absolute, so it runs from any working directory:

    python Chapter09/deploy_deps.py

CMake invokes it while configuring `Chapter09/viewer`, so a normal build needs no separate step.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.join(os.path.dirname(HERE), "deps")

# The exit status is passed on rather than dropped: the upstream wrapper uses `os.system` and discards its result, so a failed bootstrap
# exits 0, CMake's COMMAND_ERROR_IS_FATAL never fires, and a truncated download surfaces much later as a confusing build error.
sys.exit(subprocess.call([
    sys.executable,
    os.path.join(DEPS, "bootstrap.py"),
    "-b", DEPS,
    "--bootstrap-file=" + os.path.join(DEPS, "bootstrap.json"),
    "--break-on-first-error",
]))
