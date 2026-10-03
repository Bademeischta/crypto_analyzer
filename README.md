# Crypto Analyzer

KI-gestützte Krypto- & Memecoin-Analyse mit technischen Indikatoren, Sentiment-Daten und einem
**ehrlich evaluierten** Machine-Learning-Modell. Nur kostenlose APIs, kein API-Key nötig.

> ⚠️ **Diese Anwendung dient ausschließlich Bildungszwecken und stellt keine Finanzberatung dar.
> Krypto-Märkte sind hochspekulativ. Handle niemals auf Basis von KI-Vorhersagen allein.
> Vergangene Performance – auch im Backtest – garantiert keine zukünftigen Ergebnisse.**

---

## Schnellstart (Windows)

```cmd
cd C:\Users\silas\OneDrive\Dokumente\crypto_analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`python main.py` startet das Dashboard unter `http://localhost:8501` (gleichbedeutend mit
`streamlit run main.py`).

### Kommandozeile

```cmd
python main.py check                     :: Pakete & API-Erreichbarkeit prüfen
python main.py analyze PEPE -i 4h        :: Analyse im Terminal (--retrain, --json)
python main.py scan --top 10             :: Signal-Scanner über die liquidesten Coins
python main.py clear-cache               :: API-Cache löschen
```

Voraussetzungen: Python 3.11+, Internetverbindung (Binance, CoinGecko, alternative.me, Reddit).

---

## Google Colab (ohne Installation)

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bademeischta/crypto_analyzer/blob/v2/notebooks/crypto_analyzer_colab.ipynb)

Link öffnen und die Zellen von oben nach unten ausführen. Das Notebook
[`notebooks/crypto_analyzer_colab.ipynb`](notebooks/crypto_analyzer_colab.ipynb) bietet Analyse,
Modellqualität, Backtest, Coin-Vergleich und Scanner direkt als Zellen mit Formularfeldern – und startet auf Wunsch
das komplette Dashboard über einen Cloudflare-Tunnel (öffentlicher `…trycloudflare.com`-Link, kein Account nötig).

- Colab-Laufzeiten sind flüchtig: Cache und Modelle optional in Google Drive speichern (Häkchen in Schritt 1).
- Colab-Server stehen meist in den USA, wo `api.binance.com` gesperrt ist – die App nutzt dann automatisch den
  offiziellen Mirror `data-api.binance.vision`.
- CoinGecko teilt sein kostenloses Rate-Limit mit allen Colab-Nutzern; Market Cap & Rang können daher fehlen.

---

## Funktionen

| Tab | Inhalt |
|---|---|
| **📈 Chart & Signal** | Candlesticks mit EMA 9/21/50/200, Bollinger Bands, Volumen, optional historische Out-of-Sample-Signale · KI-Einschätzung mit Wahrscheinlichkeiten, coin-spezifischen Schwellen und erwarteter Volatilität |
| **🧮 Indikatoren** | RSI, MACD, ADX/±DI, Stochastic + Tabelle mit Einordnung der aktuellen Werte |
| **🤖 KI-Modell** | Log-Loss-Skill, p-Wert, Trefferquote vs. Baseline, MCC, Verlauf je Testperiode, Kalibrierungsdiagramm, Konfusionsmatrix, „Was geschah nach Signalen?", Permutation-Feature-Importance, Modell-Steckbrief |
| **💰 Backtest** | Interaktiver Backtest der Out-of-Sample-Signale (Schwelle, Kosten, Haltedauer, Short) vs. Buy & Hold mit Equity- und Drawdown-Kurve |
| **💭 Sentiment** | Fear & Greed Index (Tacho + Verlauf), CoinGecko-Community-Stimmung, Reddit-Erwähnungen |
| **⚖️ Vergleich** | Relative Performance, Rendite/Volatilität/Sharpe/Drawdown/Beta, Korrelationsmatrix für bis zu 8 Coins |
| **🔭 Scanner** | Signale + gemessene Modellqualität für die liquidesten Coins auf einen Blick |

Coin-Auswahl: Top-150 nach Binance-Volumen **oder beliebiges Symbol eintippen** (z.B. `WIF`, `BONK`).
Intervalle: 1h, 4h, 1d.

---

## Wie das Modell funktioniert

```
Binance-Kerzen ─┐
BTC/ETH-Kerzen ─┼─► Validierung ─► 35–38 stationäre Features ─► Labels ─► Purged Walk-Forward ─► Kalibrierung ─► Signal
Fear & Greed  ──┘   (Lücken,       (Oszillatoren, Ratios,       (vola-    (20 Testperioden,      (nur mit
                    OHLC-Fehler)   Korrelation/Beta, F&G)       adj.)     Embargo = Horizont)    früheren Folds)
```

**Keine Zukunftsdaten, nirgends.** Das ist automatisiert getestet (`tests/test_pipeline.py`): Hängt man
zukünftige Kerzen an, darf sich kein einziger vergangener Feature-Wert und kein Label ändern.

- **Features** sind stationär (Verhältnisse statt Preisniveaus – per Test: ein 1000× höherer Preis ändert kein
  Feature). Dazu Orderflow (Taker-Buy-Anteil), Garman-Klass-Volatilität, Efficiency Ratio, Korrelation/Beta/relative
  Stärke zu BTC & ETH und die Fear-&-Greed-**Historie** (um einen Tag verzögert).
- **Labels:** BULLISH/BEARISH, wenn der Kurs in 5 Tagen um mehr als `0,4 · σ · √h` steigt/fällt. Die Schwelle passt
  sich der aktuellen Volatilität an – für BTC z.B. ±2 %, für PEPE ±6 %. Das Volatilitätsregime wird gegen Quantile
  der *vergangenen* Volatilität klassifiziert.
- **Purged Walk-Forward:** Bis zu 20 aufeinanderfolgende Testperioden, verankert an den neuesten Daten. Zwischen
  Training und Test liegt ein Embargo in Länge des Vorhersage-Horizonts, weil die letzten Trainingslabels sonst
  schon Kurse aus der Testperiode "kennen".
- **Kalibrierung:** Boosting-Modelle sind auf verrauschten Daten massiv überkonfident (gemessen: „61 % sicher" →
  39 % Treffer). Jede Testperiode wird mit Temperature Scaling + Mischung mit den Klassenhäufigkeiten korrigiert –
  gelernt ausschließlich auf früheren Testperioden. Ein Modell ohne Information fällt so automatisch auf die
  Klassenhäufigkeiten zurück, statt falsche Signale zu erzeugen.
- **Bewertung:** Hauptkriterium ist der Log-Loss-Skill gegenüber reinem Raten nach Klassenhäufigkeit, mit
  blockweisem t-Test (überlappende Labels ⇒ nur n/h unabhängige Stichproben).
- **Engine:** LightGBM, falls installiert, sonst scikit-learn HistGradientBoosting (gleiches Verfahren, langsamer).
  Walk-Forward-Folds werden parallel auf allen CPU-Kernen trainiert.

---

## Ehrliche Einschätzung – gemessen, nicht behauptet

Out-of-Sample-Ergebnisse der Standard-Konfiguration (September 2026; 8 Coins: BTC, ETH, SOL, XRP, BNB, DOGE, ADA,
PEPE; 5-Tage-Horizont; 900 bzw. 3600 OOS-Vorhersagen je Coin):

| | Tageskerzen (1d) | 4h-Kerzen |
|---|---|---|
| **Kursrichtung:** signifikant besser als Raten | 0 von 8 | 0 von 8 |
| Ø Log-Loss-Skill Richtung | −0,4 % | −0,4 % |
| **Volatilitätsregime:** signifikant besser als Raten | 3 von 8 (+4 schwach) | **7 von 8** |
| Ø Trefferquote Volatilität vs. Baseline | 40 % vs. 34 % | **48 % vs. 34 %** |

**Was das bedeutet:**

- Die **Kursrichtung** der nächsten Tage lässt sich aus Indikatoren, Sentiment und Marktstruktur nicht verlässlich
  vorhersagen – das Modell sagt das offen und zeigt dann meist „Kein klares Signal". Tools, die hier 70 %+ Trefferquote
  versprechen, testen fast immer mit Zukunftsdaten.
- Die **Volatilität** (ruhige vs. hektische Phase) ist dagegen messbar vorhersagbar (Volatility Clustering) – nützlich
  für Positionsgrößen, Stop-Abstände und Risikomanagement.
- Diese Einschätzung wird für **jeden Coin und jedes Intervall live neu gemessen** und im Tab „KI-Modell" angezeigt.

---

## Architektur

```
crypto_analyzer/
├── config.yaml            ← Alle Einstellungen (Werte auch pro Intervall möglich)
├── main.py                ← Dashboard-Start + CLI (check / analyze / scan / clear-cache)
├── src/
│   ├── config.py          ← Laden/Validieren, Logging, Zeitrahmen-Umrechnung (Tage ↔ Kerzen)
│   ├── data/
│   │   ├── http.py        ← Session-Pooling, Retry mit Jitter, Retry-After, Rate-Limiting pro Host
│   │   ├── cache.py       ← Atomarer Disk-Cache (JSON + DataFrames), Stale-Fallback
│   │   ├── fetcher.py     ← Binance (inkrementell, parallel, Endpoint-Fallback), CoinGecko
│   │   └── validator.py   ← Lückenerkennung, OHLC-Reparatur, Handelspausen, Ausreißer
│   ├── features/
│   │   ├── technical.py   ← Indikatoren (Wilder-korrekt) + stationäre Features
│   │   ├── sentiment.py   ← Fear & Greed (Historie), Reddit (JSON → RSS-Fallback, Circuit-Breaker)
│   │   └── pipeline.py    ← Feature-Matrix, Marktstruktur, vola-adjustierte Labels
│   ├── models/
│   │   ├── engine.py      ← LightGBM/HistGB, Klassen-Alignment, Kalibrierung
│   │   ├── trainer.py     ← Purged Walk-Forward (parallel), Permutation-Importance, Persistenz
│   │   ├── evaluator.py   ← Log-Loss-Skill, Signifikanz, Kalibrierung, Signalstatistik
│   │   ├── backtest.py    ← Vektorisierter Backtest inkl. Kosten
│   │   └── predictor.py   ← Live-Signal mit Kontext
│   ├── analysis/
│   │   ├── analyzer.py    ← Orchestrierung, Vergleich, Scanner
│   │   └── report.py      ← Text-Zusammenfassung (CLI & Notebook)
│   └── ui/                ← Streamlit-Dashboard + Plotly-Komponenten (Light/Dark)
├── notebooks/             ← Google-Colab-Notebook
└── tests/                 ← 95 Offline-Tests (pytest), u.a. Lookahead-, Stationaritäts-, Backtest-Tests
```

**Datenquellen** (alle kostenlos, kein Key):
[Binance Public API](https://developers.binance.com/docs/binance-spot-api-docs) ·
[CoinGecko](https://www.coingecko.com/en/api) · [alternative.me Fear & Greed](https://alternative.me/crypto/fear-and-greed-index/) ·
Reddit (öffentliche Feeds)

**Performance:** Kursdaten werden inkrementell gecacht (Refresh = 1 Request für die neuen Kerzen), alle Quellen
parallel geladen, Folds parallel trainiert. Typisch: erste Analyse eines Coins 2–5 s, danach < 0,5 s.

---

## Konfiguration

Alles in `config.yaml` – kein Code-Edit nötig. Beispiele:

```yaml
ml:
  direction:
    horizon_days: 5            # Vorhersage-Horizont
    vol_multiplier: 0.4        # Schwelle = 0,4 · σ · √h
  confidence_display_threshold: 0.5
  training_lookback_days: { "1h": 365, "4h": 1095, default: 1825 }
backtest:
  fee_bps: 10                  # 0,1 % pro Seite
```

Änderungen an `features` oder `ml` invalidieren gespeicherte Modelle automatisch (Config-Fingerprint).

---

## Entwicklung

```cmd
pip install -r requirements-dev.txt
python -m pytest          :: 95 Tests, komplett offline, ~7 s
python -m ruff check src tests main.py
```

GitHub Actions testet auf Linux & Windows, Python 3.11/3.12, mit LightGBM und mit dem scikit-learn-Fallback.

---

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` (im aktivierten venv) |
| LightGBM lässt sich nicht installieren | Zeile aus `requirements.txt` entfernen – die App nutzt automatisch scikit-learn |
| `Symbol nicht gefunden` | Symbol ohne USDT eingeben (BTC, nicht BTCUSDT); der Coin muss als USDT-Paar auf Binance gehandelt werden |
| `HTTP 451/403` von Binance | Wird automatisch über `data-api.binance.vision` umgangen |
| Marktdaten „von vor X Min." | CoinGecko-Rate-Limit – der letzte bekannte Stand wird angezeigt |
| „Zu wenig Historie" | Coin ist zu neu für Tageskerzen → 4h oder 1h wählen (mehr Kerzen) |
| `python main.py check` | Prüft Pakete und API-Erreichbarkeit |
