"""Entry Point für den Crypto Analyzer.

Aufruf: streamlit run main.py

Unterstützt Python 3.10+ (inkl. Google Colab).
CLI-Check: python main.py --check
"""

from __future__ import annotations

import sys
from pathlib import Path


def _check_python_version() -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) < (3, 10):
        print(
            f"FEHLER: Python 3.10+ erforderlich. "
            f"Installierte Version: {major}.{minor}.\n"
            f"Bitte installiere Python 3.10 oder neuer von https://python.org"
        )
        sys.exit(1)


def _check_config() -> Path:
    root = Path(__file__).parent
    config_path = root / "config.yaml"
    if not config_path.exists():
        print(
            "FEHLER: config.yaml nicht gefunden.\n"
            f"Erwartet in: {config_path}\n"
            "Stelle sicher dass du im Projekt-Verzeichnis bist."
        )
        sys.exit(1)
    return config_path


def _cli_check() -> None:
    print("Crypto Analyzer v2 – Dependency Check")
    print("=" * 40)

    checks = [
        ("Python 3.10+", lambda: sys.version_info >= (3, 10)),
        ("streamlit", lambda: __import__("streamlit")),
        ("pandas", lambda: __import__("pandas")),
        ("numpy", lambda: __import__("numpy")),
        ("plotly", lambda: __import__("plotly")),
        ("lightgbm", lambda: __import__("lightgbm")),
        ("scikit-learn", lambda: __import__("sklearn")),
        ("joblib", lambda: __import__("joblib")),
        ("PyYAML", lambda: __import__("yaml")),
        ("requests", lambda: __import__("requests")),
        ("aiohttp", lambda: __import__("aiohttp")),
        ("rich", lambda: __import__("rich")),
        ("pyngrok (optional)", lambda: __import__("pyngrok")),
    ]

    all_ok = True
    for name, check in checks:
        try:
            check()
            print(f"  OK  {name}")
        except Exception as exc:
            if "optional" in name:
                print(f"  --  {name}: nicht installiert (nur für Colab nötig)")
            else:
                print(f"  FEHLT  {name}: {exc}")
                all_ok = False

    print("=" * 40)
    if all_ok:
        print("Alle Dependencies vorhanden. Starte mit: streamlit run main.py")
    else:
        print("Fehlende Packages: pip install -r requirements.txt")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
_check_python_version()

if len(sys.argv) > 1 and sys.argv[1] == "--check":
    _cli_check()
    sys.exit(0)

_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_check_config()

from src.ui.dashboard import render_dashboard

render_dashboard()
