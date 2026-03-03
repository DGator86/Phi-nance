"""
Species Registry — 28 fixed leaf nodes of the KPCOFGS taxonomy.

Each species entry defines:
  - Taxonomic path (kingdom, phylum, class_, order, family, genus)
  - base_regime : which of the 8 collapsed projection bins this species feeds
                  ('TREND', 'RANGE', 'BREAKOUT', 'EXHAUST_REV')
                  TREND and BREAKOUT are split TREND_UP/DN and BREAKOUT_UP/DN
                  at runtime via drift sign.
  - phylum_regime: 'LOWVOL' | 'HIGHVOL' | None  (secondary regime tag)

The 28 species are mutually exclusive and collectively exhaustive within
the defined taxonomy.  Do NOT expand beyond 28 without versioning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Species:
    id: str                    # S01 … S28
    kingdom: str               # DIR | NDR | TRN
    phylum: str                # LV  | NV  | HV
    class_: str                # PT PX TE | BR RR AR | SR RB FB
    order: str                 # AGC | RVP | ABS | EXH
    family: str                # ALN | CT  | CST
    genus: str                 # RUN | PBM | FLG | VWM | RRO | SRR
    description: str
    base_regime: str           # TREND | RANGE | BREAKOUT | EXHAUST_REV
    phylum_regime: Optional[str] = None  # LOWVOL | HIGHVOL | None


# ──────────────────────────────────────────────────────────────────────────────
# 28 species definitions
# ──────────────────────────────────────────────────────────────────────────────

SPECIES_LIST: List[Species] = [

    # ═══ DIR + LV ═══════════════════════════════════════════════════════════
    Species(
        id="S01", kingdom="DIR", phylum="LV", class_="PT",
        order="AGC", family="ALN", genus="RUN",
        description="Low-vol persistent trend — aligned run",
        base_regime="TREND", phylum_regime="LOWVOL",
    ),
    Species(
        id="S02", kingdom="DIR", phylum="LV", class_="PT",
        order="RVP", family="CST", genus="PBM",
        description="Low-vol persistent trend — constrained pullback",
        base_regime="TREND", phylum_regime="LOWVOL",
    ),
    Species(
        id="S03", kingdom="DIR", phylum="LV", class_="TE",
        order="EXH", family="CT", genus="RRO",
        description="Low-vol trend exhaustion — oscillatory reversion",
        base_regime="EXHAUST_REV", phylum_regime="LOWVOL",
    ),

    # ═══ DIR + NV ═══════════════════════════════════════════════════════════
    Species(
        id="S04", kingdom="DIR", phylum="NV", class_="PT",
        order="AGC", family="ALN", genus="RUN",
        description="Normal-vol persistent trend — aligned run",
        base_regime="TREND", phylum_regime=None,
    ),
    Species(
        id="S05", kingdom="DIR", phylum="NV", class_="PT",
        order="RVP", family="ALN", genus="PBM",
        description="Normal-vol persistent trend — pullback continuation",
        base_regime="TREND", phylum_regime=None,
    ),
    Species(
        id="S06", kingdom="DIR", phylum="NV", class_="PX",
        order="AGC", family="ALN", genus="RUN",
        description="Normal-vol expansion trend — aggressive run",
        base_regime="TREND", phylum_regime=None,
    ),
    Species(
        id="S07", kingdom="DIR", phylum="NV", class_="PX",
        order="RVP", family="ALN", genus="FLG",
        description="Normal-vol expansion trend — flag consolidation",
        base_regime="TREND", phylum_regime=None,
    ),
    Species(
        id="S08", kingdom="DIR", phylum="NV", class_="TE",
        order="EXH", family="CT", genus="RRO",
        description="Normal-vol trend exhaustion — range reversion",
        base_regime="EXHAUST_REV", phylum_regime=None,
    ),

    # ═══ DIR + HV ═══════════════════════════════════════════════════════════
    Species(
        id="S09", kingdom="DIR", phylum="HV", class_="PX",
        order="AGC", family="ALN", genus="RUN",
        description="High-vol expansion — aggressive momentum run",
        base_regime="TREND", phylum_regime="HIGHVOL",
    ),
    Species(
        id="S10", kingdom="DIR", phylum="HV", class_="PX",
        order="ABS", family="CT", genus="FLG",
        description="High-vol expansion — absorbed flag",
        base_regime="TREND", phylum_regime="HIGHVOL",
    ),
    Species(
        id="S11", kingdom="DIR", phylum="HV", class_="TE",
        order="EXH", family="CT", genus="SRR",
        description="High-vol trend exhaustion — sharp spike reversal",
        base_regime="EXHAUST_REV", phylum_regime="HIGHVOL",
    ),

    # ═══ NDR + LV ═══════════════════════════════════════════════════════════
    Species(
        id="S12", kingdom="NDR", phylum="LV", class_="BR",
        order="ABS", family="ALN", genus="VWM",
        description="Low-vol balanced range — VWAP-anchored drift",
        base_regime="RANGE", phylum_regime="LOWVOL",
    ),
    Species(
        id="S13", kingdom="NDR", phylum="LV", class_="AR",
        order="ABS", family="CT", genus="RRO",
        description="Low-vol accumulation range — oscillatory reversion",
        base_regime="RANGE", phylum_regime="LOWVOL",
    ),
    Species(
        id="S14", kingdom="NDR", phylum="LV", class_="BR",
        order="AGC", family="ALN", genus="RRO",
        description="Low-vol balanced range — mean reversion oscillation",
        base_regime="RANGE", phylum_regime="LOWVOL",
    ),

    # ═══ NDR + NV ═══════════════════════════════════════════════════════════
    Species(
        id="S15", kingdom="NDR", phylum="NV", class_="BR",
        order="ABS", family="ALN", genus="RRO",
        description="Normal-vol balanced range — absorption reversion",
        base_regime="RANGE", phylum_regime=None,
    ),
    Species(
        id="S16", kingdom="NDR", phylum="NV", class_="RR",
        order="RVP", family="CT", genus="SRR",
        description="Normal-vol reactive reversion — sharp mean return",
        base_regime="RANGE", phylum_regime=None,
    ),
    Species(
        id="S17", kingdom="NDR", phylum="NV", class_="AR",
        order="ABS", family="CT", genus="VWM",
        description="Normal-vol accumulation/distribution — volume-weighted",
        base_regime="RANGE", phylum_regime=None,
    ),
    Species(
        id="S18", kingdom="NDR", phylum="NV", class_="BR",
        order="RVP", family="CST", genus="PBM",
        description="Normal-vol balanced range — constrained mid-range pullback",
        base_regime="RANGE", phylum_regime=None,
    ),

    # ═══ NDR + HV ═══════════════════════════════════════════════════════════
    Species(
        id="S19", kingdom="NDR", phylum="HV", class_="RR",
        order="RVP", family="CT", genus="SRR",
        description="High-vol reactive reversion — sharp spike return",
        base_regime="RANGE", phylum_regime="HIGHVOL",
    ),
    Species(
        id="S20", kingdom="NDR", phylum="HV", class_="BR",
        order="ABS", family="CST", genus="RRO",
        description="High-vol compressed balanced range — choppy absorption",
        base_regime="RANGE", phylum_regime="HIGHVOL",
    ),
    Species(
        id="S21", kingdom="NDR", phylum="HV", class_="AR",
        order="EXH", family="CT", genus="SRR",
        description="High-vol distribution exhaustion — reversal",
        base_regime="EXHAUST_REV", phylum_regime="HIGHVOL",
    ),

    # ═══ TRN + LV ═══════════════════════════════════════════════════════════
    Species(
        id="S22", kingdom="TRN", phylum="LV", class_="SR",
        order="AGC", family="ALN", genus="FLG",
        description="Low-vol squeeze release — flag formation",
        base_regime="BREAKOUT", phylum_regime="LOWVOL",
    ),
    Species(
        id="S23", kingdom="TRN", phylum="LV", class_="SR",
        order="RVP", family="CST", genus="PBM",
        description="Low-vol squeeze — constrained pullback pre-release",
        base_regime="BREAKOUT", phylum_regime="LOWVOL",
    ),

    # ═══ TRN + NV ═══════════════════════════════════════════════════════════
    Species(
        id="S24", kingdom="TRN", phylum="NV", class_="SR",
        order="AGC", family="ALN", genus="RUN",
        description="Normal-vol squeeze release — directional run",
        base_regime="BREAKOUT", phylum_regime=None,
    ),
    Species(
        id="S25", kingdom="TRN", phylum="NV", class_="RB",
        order="AGC", family="ALN", genus="RUN",
        description="Normal-vol range break — momentum continuation",
        base_regime="BREAKOUT", phylum_regime=None,
    ),
    Species(
        id="S26", kingdom="TRN", phylum="NV", class_="FB",
        order="EXH", family="CT", genus="SRR",
        description="Normal-vol failed break — sharp reversal",
        base_regime="EXHAUST_REV", phylum_regime=None,
    ),

    # ═══ TRN + HV ═══════════════════════════════════════════════════════════
    Species(
        id="S27", kingdom="TRN", phylum="HV", class_="SR",
        order="AGC", family="ALN", genus="RUN",
        description="High-vol squeeze — explosive momentum release",
        base_regime="BREAKOUT", phylum_regime="HIGHVOL",
    ),
    Species(
        id="S28", kingdom="TRN", phylum="HV", class_="RB",
        order="ABS", family="ALN", genus="FLG",
        description="High-vol range break — absorbed flag continuation",
        base_regime="BREAKOUT", phylum_regime="HIGHVOL",
    ),
]

assert len(SPECIES_LIST) == 28, f"Expected 28 species, got {len(SPECIES_LIST)}"

# ──────────────────────────────────────────────────────────────────────────────
# Convenience lookups
# ──────────────────────────────────────────────────────────────────────────────

SPECIES_BY_ID: dict[str, Species] = {s.id: s for s in SPECIES_LIST}

# Which species belong to each kingdom
KINGDOM_SPECIES: dict[str, list[str]] = {k: [] for k in ("DIR", "NDR", "TRN")}
for _s in SPECIES_LIST:
    KINGDOM_SPECIES[_s.kingdom].append(_s.id)

# Class-to-kingdom parent mapping
CLASS_KINGDOM: dict[str, str] = {
    "PT": "DIR", "PX": "DIR", "TE": "DIR",
    "BR": "NDR", "RR": "NDR", "AR": "NDR",
    "SR": "TRN", "RB": "TRN", "FB": "TRN",
}

# Siblings within each kingdom's class group
CLASS_SIBLINGS: dict[str, list[str]] = {
    "DIR": ["PT", "PX", "TE"],
    "NDR": ["BR", "RR", "AR"],
    "TRN": ["SR", "RB", "FB"],
}

# 8 collapsed regime bins (runtime direction split applied in probability_field)
REGIME_BINS = [
    "TREND_UP", "TREND_DN",
    "RANGE",
    "BREAKOUT_UP", "BREAKOUT_DN",
    "EXHAUST_REV",
    "LOWVOL", "HIGHVOL",
]
