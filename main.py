"""Entry Point für den Crypto Analyzer.

Aufruf: streamlit run main.py

Prüft Python-Version und lädt das Streamlit-Dashboard.
Kann auch als CLI-Check verwendet werden: python main.py --check
"""

from __future__ import annotations

import sys
from pathlib import Path


def _check_python_version() -> None:
    """Stellt sicher dass Python 3.11+ verwendet wird."""
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) < (3, 11):
        print(
            f"FEHLER: Python 3.11+ erforderlich. "
            f"Installierte Version: {major}.{minor}.\n"
            f"Bitte installiere Python 3.11 oder neuer von https://python.org"
        )
        sys.exit(1)


def _check_config() -> Path:
    """Prüft ob config.yaml im Projektverzeichnis vorhanden ist."""
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
    """Führt Dependency-Checks durch und gibt Status aus."""
    print("Crypto Analyzer – Dependency Check")
    print("=" * 40)

    checks = [
        ("Python 3.11+", lambda: sys.version_info >= (3, 11)),
        ("streamlit", lambda: __import__("streamlit")),
        ("pandas", lambda: __import__("pandas")),
        ("numpy", lambda: __import__("numpy")),
        ("ta (Indikatoren)", lambda: __import__("ta")),
        ("plotly", lambda: __import__("plotly")),
        ("lightgbm", lambda: __import__("lightgbm")),
        ("scikit-learn", lambda: __import__("sklearn")),
        ("joblib", lambda: __import__("joblib")),
        ("PyYAML", lambda: __import__("yaml")),
        ("requests", lambda: __import__("requests")),
        ("aiohttp", lambda: __import__("aiohttp")),
        ("rich", lambda: __import__("rich")),
    ]

    all_ok = True
    for name, check in checks:
        try:
            check()
            print(f"  ✅ {name}")
        except Exception as exc:
            print(f"  ❌ {name}: {exc}")
            all_ok = False

    print("=" * 40)
    if all_ok:
        print("Alle Dependencies vorhanden. Starte mit: streamlit run main.py")
    else:
        print("Fehlende Packages installieren mit: pip install -r requirements.txt")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

_check_python_version()

# CLI-Modus: python main.py --check
if len(sys.argv) > 1 and sys.argv[1] == "--check":
    _cli_check()
    sys.exit(0)

# Streamlit-Modus: Projekt-Root in sys.path eintragen
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Konfiguration prüfen
_check_config()

# Dashboard importieren und rendern
# Dieser Import muss am Ende stehen, damit Streamlit korrekt initialisiert wird
from src.ui.dashboard import render_dashboard

render_dashboard()
