# --------------------------------------------------------------------------- #
# profiles.py — Named subsets ("profiles") of decoding methods
#
# A profile is a curated list of method names that can be selected as a group
# (e.g. "dubois_full" runs all 36 Dubois sampling strategies). Profiles let
# the CLI and experiment configs reference a bundle instead of listing methods
# individually. Each method name must match a key in generative.py or
# publication.py's method registries.
#
# "Lanes" control where results appear: main paper vs. appendix vs. experimental.
# --------------------------------------------------------------------------- #
from __future__ import annotations

from typing import Final

# A representative subset of Dubois et al. (2025) — one low + one high setting
# per parameter family, for quick experiments.
DUBOIS_REPRESENTATIVE: Final[list[str]] = [
    "ANCESTRAL",
    "TEMP_09",
    "TEMP_13",
    "REP_105",
    "REP_130",
    "TOP_K_50",
    "TOP_K_1000",
    "TOP_P_095",
    "TOP_P_03",
    "TYPICAL_095",
    "TYPICAL_03",
    "ETA_1E4",
    "ETA_10",
]

# Complete Dubois sweep — every temperature, top-k, top-p, typical-p, eta,
# and repetition-penalty setting from the paper.
DUBOIS_FULL: Final[list[str]] = [
    "ANCESTRAL",
    "TEMP_05",
    "TEMP_07",
    "TEMP_09",
    "TEMP_11",
    "TEMP_12",
    "TEMP_13",
    "REP_105",
    "REP_110",
    "REP_115",
    "REP_120",
    "REP_125",
    "REP_130",
    "TOP_K_10",
    "TOP_K_20",
    "TOP_K_50",
    "TOP_K_75",
    "TOP_K_100",
    "TOP_K_1000",
    "TOP_P_03",
    "TOP_P_05",
    "TOP_P_07",
    "TOP_P_08",
    "TOP_P_09",
    "TOP_P_095",
    "TYPICAL_03",
    "TYPICAL_05",
    "TYPICAL_07",
    "TYPICAL_08",
    "TYPICAL_09",
    "TYPICAL_095",
    "ETA_1E4",
    "ETA_5E3",
    "ETA_1E3",
    "ETA_01",
    "ETA_05",
    "ETA_10",
]

# Methods not in the original Dubois set — our novel contributions
NOVEL_CORE: Final[list[str]] = [
    "CONTRASTIVE_K8_A06",
    "CFG_20",
    "P_LESS",
    "TOP_H_05",
    "TOP_H_07",
    "MBR_16_BERTSCORE",
]

PUBLICATION_CORE: Final[list[str]] = [*DUBOIS_REPRESENTATIVE, *NOVEL_CORE]

# Full replication + extension: all 37 Dubois configs plus our novel methods
PUBLICATION_FULL: Final[list[str]] = [*DUBOIS_FULL, *NOVEL_CORE]

# Structural / search-based methods still under evaluation
EXPERIMENTAL_STRUCTURAL: Final[list[str]] = [
    "FREE_DEFAULT",
    "MCTS_RM_UCT20_D16",
    "RAEE_GEN_DEFAULT",
]

METHOD_PROFILES: Final[dict[str, list[str]]] = {
    "dubois_representative": DUBOIS_REPRESENTATIVE,
    "dubois_full": DUBOIS_FULL,
    "novel_core": NOVEL_CORE,
    "publication_core": PUBLICATION_CORE,
    "publication_full": PUBLICATION_FULL,
}


def available_method_profiles() -> list[str]:
    return sorted(METHOD_PROFILES)


def resolve_method_profile(name: str | None) -> list[str]:
    if not name:
        return []
    try:
        return list(METHOD_PROFILES[name])
    except KeyError as exc:
        raise KeyError(f"Unknown method profile: {name}") from exc


# Determines where a method's results appear in the paper output
def publication_lane_for_method(method_name: str) -> str:
    if method_name in PUBLICATION_CORE:
        return "publication"
    if method_name in DUBOIS_FULL:
        return "appendix"
    if method_name in EXPERIMENTAL_STRUCTURAL:
        return "experimental"
    return "legacy"
