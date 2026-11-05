import numpy as np
import pytest

from simulazione_riscaldamento import HOUSE_PRESETS, HeatingSimulation


def _sim(profile_key="standard", **kwargs):
  params = {"house": HOUSE_PRESETS[profile_key]}
  params.update(kwargs)
  return HeatingSimulation(**params)


@pytest.mark.parametrize("profile_key", list(HOUSE_PRESETS))
def test_steady_state_air_at_target(profile_key):
  sim = _sim(profile_key)
  t_air, t_wall = sim._steady_state_temperatures()
  assert t_air == pytest.approx(sim.t_target)
  assert sim.t_outside < t_wall < t_air


@pytest.mark.parametrize("profile_key", list(HOUSE_PRESETS))
def test_always_on_keeps_air_near_target(profile_key):
  res = _sim(profile_key).run(strategy="always_on")
  assert np.min(res["t_air"]) > 20.0
  assert np.max(res["t_air"]) <= 21.0 + 1e-9


@pytest.mark.parametrize("profile_key", list(HOUSE_PRESETS))
def test_off_while_out_saves_energy_and_cools_down(profile_key):
  sim = _sim(profile_key)
  res_on = sim.run(strategy="always_on")
  res_off = sim.run(strategy="off_while_out")

  on_kwh = res_on["energy_kwh"][-1]
  off_kwh = res_off["energy_kwh"][-1]
  assert 0.0 < off_kwh < on_kwh

  # Durante la finestra fuori casa il riscaldatore è spento
  away = (res_off["time_min"] >= sim.away_start_min) & (
      res_off["time_min"] < sim.reheat_start_min
  )
  assert np.all(res_off["heater_kw"][away] == 0.0)

  # La temperatura dell'aria scende sotto il target e torna a regime a fine giornata
  assert np.min(res_off["t_air"]) < sim.t_target - 1.0
  assert res_off["t_air"][-1] == pytest.approx(sim.t_target, abs=1.0)


def test_unknown_strategy_raises():
  with pytest.raises(ValueError, match="Strategia sconosciuta"):
    _sim().run(strategy="modalita_sconosciuta")


def test_simulation_runs_24_hours():
  res = _sim().run(strategy="always_on")
  for key in ("time_min", "t_air", "t_wall", "heater_kw", "leak_kw", "energy_kwh"):
    assert len(res[key]) == 24 * 60
  assert np.all(np.isfinite(res["t_air"]))
