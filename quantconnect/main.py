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
        self._phi_bar_count = 0
        self._desired_mode = None  # "long" | "flat" — from PHI; applied on schedule
        self._last_phi_close = 0.0
        self._warned_no_px = False

        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.phi = self.AddData(PhiNanceOHLCV, "PHI_SPY", Resolution.Daily).Symbol

        # OnEndOfDay(symbol) is easy to mis-wire in Python (symbol != self.spy by identity).
        # Schedule is explicit and matches QC docs for EOD rebalance.
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 1),
            self._apply_target_position,
        )

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
            return

        custom = data[self.phi]
        self._phi_bar_count += 1
        self._last_phi_close = float(custom.Close)

        if self._use_signal_card:
            regime = str(self.regime_card.get("composite_regime", "") or "")
            playbook = str(self.regime_card.get("playbook_regime_key", "") or "")
            label = (regime or playbook).upper()
            if label.startswith("BULL"):
                self._desired_mode = "long"
            elif label.startswith("BEAR"):
                self._desired_mode = "flat"
            else:
                self._desired_mode = "flat"
        else:
            self._desired_mode = "long" if custom.Close > custom.Open else "flat"

        if self._phi_bar_count <= 3 or self._phi_bar_count % 60 == 0:
            ds = str(custom.Time.date())
            rc = (
                (self.regime_card.get("composite_regime") or "n/a")
                if self._use_signal_card
                else "bar-rule"
            )
            self.Debug(f"PHI_SPY {ds} close={custom.Close:.2f} mode={rc}")

    def _apply_target_position(self):
        """Run before SPY close; use SPY price when local equity data exists, else PHI close."""
        if self._desired_mode is None:
            return
        spy_sec = self.Securities[self.spy]
        spy_px = float(spy_sec.Price)
        phi_px = float(self._last_phi_close)
        if spy_px <= 0 and phi_px <= 0:
            if not self._warned_no_px:
                self.Debug(
                    "PhiNance: no SPY price and no PHI close; "
                    "install SPY daily under workspace data/ or sync ohlcv.csv."
                )
                self._warned_no_px = True
            return
        if spy_px > 0:
            if self._desired_mode == "long":
                self.SetHoldings(self.spy, 1.0)
            else:
                self.Liquidate(self.spy)
            return
        cur = int(self.Portfolio[self.spy].Quantity)
        cash = float(self.Portfolio.Cash)
        approx_value = cash + float(cur) * phi_px
        if self._desired_mode == "long":
            want = int((approx_value * 0.998) / phi_px)
            delta = want - cur
            if delta != 0:
                self.MarketOrder(self.spy, delta)
        elif cur != 0:
            self.Liquidate(self.spy)
