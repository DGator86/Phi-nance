# region imports
from AlgorithmImports import *
# endregion


class PhiNanceOHLCV(PythonData):
    """Custom bar type for CSV exported by Phi-nance (time,open,high,low,close,volume).

    Upload ``ohlcv.csv`` under your QC project Data folder (path below), or use Object Store.
    """

    def GetSource(self, config, date, isLiveMode):
        # TODO: replace with your Object Store URL or Data/ relative path
        return SubscriptionDataSource(
            "phi_nance/spy/ohlcv.csv",
            SubscriptionTransportMedium.LocalFile,
        )

    def Reader(self, config, line, date, isLiveMode):
        if not (line and line[0].isdigit()):
            return None
        bar = PhiNanceOHLCV()
        bar.Symbol = config.Symbol
        parts = line.split(",")
        bar.Time = datetime.strptime(parts[0], "%Y-%m-%d")
        bar.Open = float(parts[1])
        bar.High = float(parts[2])
        bar.Low = float(parts[3])
        bar.Close = float(parts[4])
        bar.Volume = float(parts[5])
        bar.Value = bar.Close
        return bar


class PhiNanceBridgeAlgorithm(QCAlgorithm):
    """Example: ingest Phi-nance CSV as custom data; trade liquid equity ``SPY``."""

    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.phi = self.AddData(PhiNanceOHLCV, "PHI_SPY", Resolution.Daily).Symbol

    def OnData(self, data):
        if not data.ContainsKey(self.phi):
            return
        custom = data[self.phi]
        # Placeholder rule — replace with logic derived from signal_card.json or Lean indicators
        if custom.Close > custom.Open:
            self.SetHoldings(self.spy, 1.0)
        else:
            self.Liquidate(self.spy)
