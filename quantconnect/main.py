# region imports
from AlgorithmImports import *
import json
import os
from pathlib import Path

from QuantConnect import Globals
# endregion


class PhiNanceOHLCV(PythonData):
    """Phi-nance export CSV: time,open,high,low,close,volume.

    **Lean CLI:** copy ``ohlcv.csv`` into the **organization workspace** ``data/``
    folder (sibling of your project dirs; see ``lean.json`` → ``data-folder``), not
    only under ``<project>/data/``. ``GetSource`` joins ``Globals.DataFolder`` per
    QuantConnect local custom-data docs.
    """

    def GetSource(self, config, date, isLiveMode):
        path = os.path.join(str(Globals.DataFolder), "ohlcv.csv").replace("\\", "/")
        return SubscriptionDataSource(path, SubscriptionTransportMedium.LocalFile)

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
    """Custom OHLCV + optional ``signal_card.json`` (single snapshot from Phi-nance export)."""

    def Initialize(self):
        # Align with typical Phi-nance daily export (first row often first trading day).
        self.SetStartDate(2023, 1, 3)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        self._logged_missing_phi = False
        self._phi_bar_count = 0

        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.phi = self.AddData(PhiNanceOHLCV, "PHI_SPY", Resolution.Daily).Symbol

        self.regime_card = self._load_signal_card()
        self._use_signal_card = bool(
            self.regime_card.get("composite_regime")
            or self.regime_card.get("playbook_regime_key")
        )

    def _load_signal_card(self):
        card_path = Path(__file__).resolve().parent / "signal_card.json"
        try:
            return json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            self.Debug(
                "Could not load signal_card.json from "
                + str(card_path)
                + " — using basic bar rule (close vs open)"
            )
            return {}

    def OnData(self, data):
        if not data.ContainsKey(self.phi):
            if not self._logged_missing_phi:
                self.Debug(
                    "No PHI_SPY custom data. Place ohlcv.csv in the workspace data/ "
                    "folder (Globals.DataFolder), e.g. LeanWorkspace/data/ohlcv.csv."
                )
                self._logged_missing_phi = True
            return

        custom = data[self.phi]
        self._phi_bar_count += 1

        if self._use_signal_card:
            regime = str(self.regime_card.get("composite_regime", "") or "")
            playbook = str(self.regime_card.get("playbook_regime_key", "") or "")
            label = (regime or playbook).upper()
            if label.startswith("BULL"):
                self.SetHoldings(self.spy, 1.0)
            elif label.startswith("BEAR"):
                self.Liquidate(self.spy)
            else:
                self.Liquidate(self.spy)
        else:
            if custom.Close > custom.Open:
                self.SetHoldings(self.spy, 1.0)
            else:
                self.Liquidate(self.spy)

        if self._phi_bar_count <= 3 or self._phi_bar_count % 60 == 0:
            ds = str(custom.Time.date())
            rc = (self.regime_card.get("composite_regime") or "n/a") if self._use_signal_card else "bar-rule"
            self.Debug(f"PHI_SPY {ds} close={custom.Close:.2f} mode={rc}")
