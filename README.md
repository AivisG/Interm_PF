Das Programm lädt Daten von yfinance, wobei der Benutzer ein Ticker-Symbol und einen Zeitraum eingeben kann.
Es werden die technischen Indikatoren RSI und MACD berechnet.
Die Daten werden verarbeitet und drei Modelle werden trainiert:
•	LSTM 
•	XGBoost
•	Gaussian Process
LSTM-Modell wird trainiert, wobei Hyperparameter-Tuning verwendet wird.
Anschließend wird das Modell mit folgenden Metriken bewertet:
•	MSE (Mean Squared Error)
•	MAE (Mean Absolute Error)
•	R² (Bestimmtheitsmaß)
•	Direction Values (Richtungskorrelation)
Die vorhergesagten Werte werden in einem Plot visualisiert, der folgende Komponenten enthält:
•	Actual Value (tatsächlicher Wert)
•	Predicted Value (vorhergesagter Wert)
•	Confidence Interval (Vertrauensintervall)
Darauf folgt ein weiterer Vergleichsgraph, der für den zukünftigen Vergleich mit anderen Modellen vorgesehen ist.
Danach wird ein Backtest durchgeführt, mit den folgenden Ergebnissen:
•	Final Balance (Endsaldo)
•	Total Return (Gesamtrendite)
•	Sharpe Ratio (Risikoadjustierte Rendite)
•	Max Drawdown (Maximaler Verlust)
Am Ende werden die Training Loss und Validation Loss als Grafik dargestellt.
Alle Diagramme werden in einer einzigen PDF-Datei gespeichert.
Der Dateiname enthält das Ticker-Symbol und den Zeitraum.
