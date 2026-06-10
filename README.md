# Crypto Analyzer

KI-gestützte Krypto- & Memecoin-Analyse mit LightGBM-Vorhersagen, technischen Indikatoren und Sentiment-Daten. Ausschließlich kostenlose APIs.

> ⚠️ **Diese Anwendung dient ausschließlich Bildungszwecken und stellt keine Finanzberatung dar. Krypto-Märkte sind hochspekulativ. Handle niemals auf Basis von KI-Vorhersagen allein. Vergangene Performance garantiert keine zukünftigen Ergebnisse.**

---

## Schnellstart (Windows CMD – Copy-Paste-Block)

```cmd
cd C:\Users\silas\OneDrive\Dokumente\crypto_analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```

Der Browser öffnet sich automatisch unter `http://localhost:8501`.

---

## Voraussetzungen

- Python 3.11 oder neuer ([Download](https://python.org/downloads))
- Internetverbindung (für Binance & CoinGecko APIs)
- ~200 MB Speicher für Dependencies

## Dependency-Check

```cmd
python main.py --check
```

---

## Funktionen

### Phase 1: Technische Analyse
- **Interaktiver Preischart** (Candlestick + EMA-Overlays + Bollinger Bands + Volumen)
- **RSI** (7 & 14) mit Überkauft/Überverkauft-Linien
- **MACD** mit Histogramm
- **Marktdaten** (Market Cap, Volumen, 24h/7d Änderung) von CoinGecko
- **Trending Coins** Sidebar

### Phase 2: KI-Vorhersagen
- **LightGBM-Klassifikation** (3 Klassen: BULLISH / NEUTRAL / BEARISH)
- **Walk-Forward-Validation** (kein Lookahead-Bias, chronologisch korrekt)
- **Konfidenz-Anzeige** (nur wenn >55% Klassenwahrscheinlichkeit)
- **Volatilitäts-Regime** (NIEDRIG / MITTEL / HOCH)
- **Modell-Transparenz**: Feature Importance, historische Accuracy, Baseline-Vergleich
- **Auto-Retrain** nach 24 Stunden

### Phase 3: Sentiment
- **Fear & Greed Index** (alternative.me) mit Tachometer-Visualisierung
- **Reddit-Sentiment** (r/CryptoCurrency, r/CryptoMarkets, r/SatoshiStreetBets)
- **Coin-Vergleich**: Relative Performance mehrerer Coins gleichzeitig

---

## Architektur

```
crypto_analyzer/
├── config.yaml          ← Alle Einstellungen hier ändern
├── main.py              ← Einstiegspunkt
├── src/
│   ├── data/            ← API-Fetching, Caching, Validierung
│   ├── features/        ← Technische Indikatoren, Sentiment, Pipeline
│   ├── models/          ← Training, Inferenz, Evaluierung
│   ├── analysis/        ← Orchestrierung
│   └── ui/              ← Streamlit Dashboard
└── data/
    ├── cache/           ← API-Cache (automatisch verwaltet)
    └── models/          ← Gespeicherte ML-Modelle
```

**Datenquellen** (alle kostenlos, kein API-Key nötig):
- [Binance Public API](https://api.binance.com) – OHLCV-Daten
- [CoinGecko Free API](https://api.coingecko.com) – Marktdaten
- [alternative.me](https://api.alternative.me/fng/) – Fear & Greed Index
- [Reddit Public JSON](https://reddit.com) – Sentiment

---

## Konfiguration

Alle Einstellungen in `config.yaml` ändern – kein Code-Edit nötig:

```yaml
ui:
  default_symbol: "BTC"      # Startmäßig angezeigter Coin
  default_lookback_days: 365  # Historische Daten

ml:
  direction:
    up_threshold: 0.02        # >2% in 5 Tagen = BULLISH
    down_threshold: -0.02     # <-2% = BEARISH
  confidence_display_threshold: 0.55  # Unter 55% = kein Signal
```

---

## Ehrliche Einschätzung der KI-Vorhersagen

Das Modell erreicht typischerweise **52-58% Accuracy** bei der Richtungsvorhersage. Das klingt wenig, ist aber bei Krypto-Märkten ein realistischer Wert – viele kommerzielle Tools behaupten höhere Werte durch Lookahead-Bias.

**Was das Modell gut kann:**
- Volatilitätsregime erkennen
- Technische Überkauft/Überverkauft-Zustände einbeziehen
- Mehrere Indikatoren gleichzeitig gewichten

**Was das Modell nicht kann:**
- News-Events, Regulierungen, Whale-Moves vorhersagen
- Memecoins mit <90 Tagen History analysieren
- Verlässliche Kursziels nennen

---

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` ausführen |
| `Symbol nicht gefunden` | Symbol ohne USDT eingeben (BTC nicht BTCUSDT) |
| `429 Rate Limit` | 1-2 Minuten warten, dann erneut versuchen |
| `Zu wenig Daten` | Lookback-Tage erhöhen (mind. 300 Tage für ML) |
| `streamlit not found` | Virtual Environment aktivieren: `venv\Scripts\activate` |
