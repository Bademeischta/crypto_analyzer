# Changelog

Alle wesentlichen Änderungen werden in dieser Datei dokumentiert.
Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

---

## [1.0.0] – 2026-05-08

### Hinzugefügt

#### Phase 1 – Fundament
- Interaktiver Candlestick-Chart mit EMA-Overlays (9, 21, 50), Bollinger Bands und Volumen (Plotly)
- RSI(7) und RSI(14) mit Overbought/Oversold-Linien
- MACD mit Signal-Linie und Histogramm
- Marktdaten von CoinGecko: Market Cap, 24h/7d Preisänderung, CMC Rank
- Trending Coins Sidebar (CoinGecko)
- Disk-Cache mit TTL für alle API-Responses (verhindert Rate-Limit-Probleme)
- Datenvalidierung mit Forward-Fill, Outlier-Erkennung, Mindest-Sample-Check
- Exponentielles Backoff bei API-Fehlern (429, 5xx)
- Coin-Schnellauswahl (BTC, ETH, SOL, DOGE, SHIB, PEPE, ...)
- Windows-11-kompatibel: pathlib.Path überall, UTF-8-Encoding explizit

#### Phase 2 – KI/ML
- LightGBM Richtungsklassifikation (BULLISH / NEUTRAL / BEARISH, 5-Tage-Horizont)
- LightGBM Volatilitätsregime-Klassifikation (NIEDRIG / MITTEL / HOCH)
- Walk-Forward-Cross-Validation (kein Lookahead-Bias, chronologisch korrekt)
- 25+ technische Features: RSI, MACD, EMA-Crosses, ADX, ATR, Bollinger Bands, OBV, Stochastic, Williams %R, Log-Returns, Korrelations-Features
- Konfidenz-Anzeige nur bei >55% Klassenwahrscheinlichkeit (ehrlicheres Signal)
- Modell-Transparenz: Feature Importance, historische Accuracy pro Fold
- Baseline-Vergleich ("immer NEUTRAL") im Dashboard
- Auto-Retrain nach 24 Stunden
- Modelle werden auf Disk gecacht (joblib)

#### Phase 3 – Sentiment & Polish
- Fear & Greed Index mit Tachometer-Visualisierung (alternative.me, kein Key nötig)
- Reddit-Sentiment: Keyword-Analyse aus r/CryptoCurrency, r/CryptoMarkets, r/SatoshiStreetBets
- Coin-Vergleichsansicht: Relative Performance (Index = 100) für bis zu 5 Coins
- Ehrlicher Disclaimer (permanent sichtbar, nicht wegklickbar)
- Datenfrisc(h)e-Anzeige (Alter der gecachten Daten)

### Technische Details
- **Sprache:** Python 3.11+
- **UI:** Streamlit ≥ 1.30.0
- **ML:** LightGBM ≥ 4.0.0 + scikit-learn ≥ 1.3.0
- **Indikatoren:** ta ≥ 0.11.0 (statt pandas-ta wegen numpy 2.0 Inkompatibilität)
- **Charts:** Plotly ≥ 5.18.0

### Bekannte Limitierungen v1.0.0
- Reddit-Sentiment ist Keyword-basiert (kein NLP/Transformer)
- Keine Unterstützung für Futures/Perpetuals (nur Spot-Preise)
- Memecoins mit <90 Tagen Handelshistorie können nicht analysiert werden
- Modell-Accuracy liegt typischerweise bei 52-58% (dokumentiert, kein Bug)

---

## [Unreleased]

### Geplant für v1.1.0
- On-Chain-Daten Integration (wenn kostenlose API verfügbar)
- Telegram-Bot Benachrichtigungen bei Signalwechsel
- Mehrsprachige UI (DE/EN)
- PDF-Export der Analyse
- Historisches Signal-Backlog (welche Signale wurden wann gegeben)

### Geplant für v2.0.0
- Prophet/NeuralProphet als optionale Modell-Alternative
- Multi-Exchange Support (Kraken, Coinbase public APIs)
- Portfolio-Tracking (mehrere Coins gleichzeitig mit Gewichtung)
