"""
GUI-free access to chartocode2's run_detection().

run_detection() lives in run_gui_v2.py, which does `import tkinter` at module top
and defines GUI classes (class X(tk.Toplevel)). A headless web server has no
tkinter/display, so we inject a lightweight stub for tkinter BEFORE importing the
module. The server never instantiates the GUI, so the stub is only needed to let
the class definitions and top-level imports succeed.

For production you may instead extract run_detection into its own GUI-free module;
this shim keeps the upstream file unmodified.
"""
import sys
import types


def _install_tk_stub():
    if "tkinter" in sys.modules:
        return
    # If real tkinter is importable (e.g. on a desktop machine), USE IT — installing
    # the stub there would leak _Dummy objects into code paths that expect real
    # values (seen as "_Dummy has no attribute 'endswith'"). Only stub when tkinter
    # is genuinely unavailable (headless server / container).
    try:
        import tkinter  # noqa: F401
        import tkinter.ttk, tkinter.filedialog, tkinter.messagebox  # noqa: F401
        return
    except Exception:
        pass

    class _Dummy:                       # generic stand-in for any tk class
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return _Dummy()
        def __call__(self, *a, **k): return _Dummy()

    def _make_module(name, attrs=None):
        m = types.ModuleType(name)
        m.__getattr__ = lambda n: _Dummy      # any attribute -> dummy class
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        return m

    tk = _make_module("tkinter", {
        "Tk": _Dummy, "Toplevel": _Dummy, "Frame": _Dummy, "Canvas": _Dummy,
        "Label": _Dummy, "Button": _Dummy, "Entry": _Dummy, "StringVar": _Dummy,
        "IntVar": _Dummy, "DoubleVar": _Dummy, "BooleanVar": _Dummy,
    })
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = _make_module("tkinter.ttk")
    sys.modules["tkinter.filedialog"] = _make_module("tkinter.filedialog")
    sys.modules["tkinter.messagebox"] = _make_module("tkinter.messagebox")


_install_tk_stub()
import run_gui_v2 as _gui                     # noqa: E402

run_detection = _gui.run_detection            # re-export the entry point
