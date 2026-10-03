# Changelog

Alle wesentlichen Änderungen werden in dieser Datei dokumentiert.
Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

---

## [2.0.0] – 2026-09-27

Komplette Überarbeitung: Die App startet wieder, das ML-Verfahren ist frei von Lookahead-Bias und wird
ehrlich out-of-sample bewertet. Dazu kommen deutlich mehr Analysefunktionen und eine automatisierte Test-Suite.

### Behoben (kritisch)
- **App startete nicht:** `ModelTrainer` las `config["ml"]["lightgbm"]`, die Config definierte aber `ml.model` → `KeyError`.
- **RSI-, MACD-, EMA- und Bollinger-Charts waren nie sichtbar:** Das UI bekam nur rohe OHLCV-Daten ohne Indikatoren.
- **Lookahead-Bias:**
  - Aktueller Fear-&-Greed-Wert wurde als Konstante auf *alle* historischen Zeilen kopiert.
  - Volatilitäts-Klassengrenzen stammten aus Quantilen über den *gesamten* Datensatz inkl. Zukunft.
  - Kein Purging/Embargo zwischen Training und Test: Trainingslabels nutzten Kurse aus der Testperiode.
- **Walk-Forward testete nie die neuesten Daten:** Folds starteten am Datenanfang und brachen nach `max_folds` ab.
- **Laufende (unfertige) Kerze** floss mit Teilvolumen und vorläufigem Schlusskurs in Features und Vorhersage ein.
- **Cache-Modelle ignorierten das Intervall:** Ein 1d-Modell wurde auf 1h-Daten angewendet.
- **Klassenspalten konnten verrutschen,** wenn in einem Trainingsblock eine Klasse fehlte.
- **Nicht-stationäre Features** (MACD/ATR in USD, kumulatives OBV) → Modell lernte Preisniveaus.
- **ATR nutzte EMA statt Wilder-Glättung**; RSI lieferte NaN statt 100 bei reinen Aufwärtsphasen.
- **Symbolauswahl-Bug:** Nach einem Klick auf einen Schnellwahl-Button ließ sich kein anderes Symbol mehr eingeben.
- **Reddit-Suche matchte Teilwörter** („sol" in „solution", „eth" in „together").
- **Lückenerkennung griff nie**, weil Binance fehlende Kerzen weglässt statt NaN zu liefern.
- **CoinGecko-Suche** nahm blind den ersten Treffer (falscher Coin bei gleichem Ticker möglich).
- `python main.py --check` schlug immer fehl (prüfte nicht benötigte Pakete `ta`/`lightgbm`).
- Annualisierung mit √252 (Aktien) statt √365 bzw. intervallabhängig.

### Hinzugefügt – Modell & Analyse
- Purged Walk-Forward mit Embargo, verankert an den neuesten Daten, bis zu 20 Testperioden, parallel trainiert.
- Volatilitätsadjustierte Richtungslabels (coin-spezifische Schwellen statt fix ±2 %).
- Zeitlich saubere Wahrscheinlichkeits-Kalibrierung (Temperature Scaling + Prior-Mischung).
- Evaluierung per Log-Loss-Skill gegen Prior-Baseline mit blockweisem Signifikanztest, dazu Kalibrierungsdiagramm,
  Konfusionsmatrix, Signalstatistik („Was geschah nach Signalen?") und Bewertung des Volatilitätsmodells.
- Permutation-Feature-Importance auf Out-of-Sample-Daten, mit lesbaren Feature-Namen.
- Neue Features: Orderflow (Taker-Buy-Anteil), Garman-Klass-Volatilität, Volatilitäts-Ratio, Efficiency Ratio,
  Abstand zu Hoch/Tief, Return-Z-Score, Beta und relative Stärke zu BTC/ETH, Fear-&-Greed-Historie.
- Backtest der Out-of-Sample-Signale inkl. Gebühren/Slippage vs. Buy & Hold (interaktiv im UI).
- Coin-Vergleich mit Korrelationsmatrix und Risiko-Kennzahlen; Markt-Scanner.
- Automatische Engine-Wahl: LightGBM oder scikit-learn HistGradientBoosting.
- Modelle werden als Bundle mit Evaluierungsdaten gespeichert (Metriken auch nach Cache-Laden sichtbar),
  invalidiert per Config-Fingerprint.

### Hinzugefügt – Daten & Infrastruktur
- Gemeinsamer HTTP-Client: Connection-Pooling, Backoff mit Jitter, `Retry-After`, Rate-Limiting pro Host.
- Inkrementeller OHLCV-Store (Refresh lädt nur neue Kerzen), parallele Batch-Requests, Binance-Endpoint-Fallback.
- Atomarer, thread-sicherer Cache mit Stale-Fallback bei API-Ausfällen.
- Validator: Reindex auf das Kerzenraster, Handelspausen-Erkennung, OHLC-Reparatur, robuste Ausreißer-Erkennung.
- Reddit: RSS-Fallback bei gesperrtem JSON-Endpoint, Circuit-Breaker bei Rate-Limits.
- CoinGecko: Community-Sentiment, ATH-Abstand, FDV; Binance-Ticker für Live-Preis.
- Alle Datenquellen einer Analyse werden parallel geladen.
- CLI: `check`, `analyze` (inkl. `--json`), `scan`, `clear-cache`; `python main.py` startet das Dashboard.
- 95 Offline-Tests (u.a. Lookahead-, Stationaritäts-, Kalibrierungs-, Backtest- und Integrationstests),
  Ruff-Linting, GitHub-Actions-CI (Linux/Windows, beide Engines).
- Logging mit rotierender Logdatei (`data/logs/`).

### Geändert
- Dashboard neu aufgebaut: Tabs, durchsuchbare Coin-Auswahl (Top-150 nach Volumen + freie Eingabe),
  Kennzahlen mit Sparkline, Light/Dark-Mode-fähige Charts.
- Training nutzt automatisch die sinnvolle Historie (bis 5 Jahre), unabhängig vom Chart-Zeitraum.
- Modell-Hyperparameter per Out-of-Sample-Vergleich gewählt (6 Coins × 2 Intervalle): stärker regularisiert,
  ohne Klassengewichte → bessere Kalibrierung bei halber Trainingszeit.
- `requirements.txt` an aktuelle Versionen angepasst (pandas 2/3, numpy 1/2); ungenutzte Pakete entfernt.

### Ergebnis der Evaluierung
Siehe README, Abschnitt „Ehrliche Einschätzung": Die Kursrichtung ist out-of-sample nicht vorhersagbar,
das Volatilitätsregime auf 4h-Basis dagegen bei 7 von 8 getesteten Coins signifikant.

---

## [1.0.0] – 2026-05-08

Erste Version: Candlestick-Chart mit Indikatoren, LightGBM-Richtungsklassifikation, Fear & Greed,
Reddit-Sentiment, Coin-Vergleich.
