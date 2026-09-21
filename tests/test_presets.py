"""Presets, Permalink-Angaben und Regler-Grenzen sind untereinander stimmig."""

import pytest

import jo_constants as C
import jo_presets as P
from jo_scenario import make_network


def test_every_preset_sets_every_control_within_bounds_and_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 5
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and p["net"] in C.NETS and p["method"] in C.METHODS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        make_network(p["net"], p["side"], p["hill"], p["eta"], p["reach"], p["spread"], p["nodes"], p["degree"], p["pot"], p["seed"])
        assert C.PRESET_HELP[name]
    assert [p["net"] for p in C.PRESETS.values()] == list(C.NETS)


def test_setting_specs_and_kept_keys_are_consistent():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(P.KEPT) <= set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        assert spec.lo is None or spec.lo < spec.hi                                 # kein Regler mit gleichen Grenzen (Streamlit bricht ab)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert "method_select" not in P.KEPT and "distance_slider" in P.KEPT            # die Umgewichtung ist immer sichtbar, das Paar der Ansicht nur bei erzeugten Netzen


def test_defaults_lie_inside_the_bounds_and_the_largest_net_has_at_most_400_nodes():
    for lo, hi, d in ((C.SIDE_MIN, C.SIDE_MAX, C.DEFAULT_SIDE), (C.HILL_MIN, C.HILL_MAX, C.DEFAULT_HILL), (C.ETA_MIN, C.ETA_MAX, C.DEFAULT_ETA), (C.REACH_MIN, C.REACH_MAX, C.DEFAULT_REACH),
                      (C.SPREAD_MIN, C.SPREAD_MAX, C.DEFAULT_SPREAD), (C.NODES_MIN, C.NODES_MAX, C.DEFAULT_NODES), (C.DEGREE_MIN, C.DEGREE_MAX, C.DEFAULT_DEGREE), (C.POT_MIN, C.POT_MAX, C.DEFAULT_POT),
                      (C.DISTANCE_MIN, C.DISTANCE_MAX, C.DEFAULT_DISTANCE)):
        assert lo < d < hi
    assert C.DEFAULT_NET == "small" and C.DEFAULT_METHOD in C.METHODS and C.SIDE_MAX ** 2 <= 400 and C.NODES_MAX <= 400


def test_choice_casters_reject_unknown_values():
    for key, good, bad in (("net_select", "ev", "ring"), ("method_select", "constant", "astar")):
        cast = P.SETTING_SPECS[key].caster
        assert cast(good) == good
        with pytest.raises(ValueError):
            cast(bad)
