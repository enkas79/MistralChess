"""Simulazione termodinamica riscaldamento: Always On vs Off While Out.

Riproduce la dinamica del video con modello a due capacità (Aria + Pareti).
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HouseProfile:
  name: str
  c_air: float  # Capacità termica aria + mobili leggeri (kWh/°C)
  c_wall: float  # Capacità termica muri / struttura (kWh/°C)
  h_air_out: float  # Dispersione diretta aria -> esterno (kW/°C) [ventilazione, finestre]
  h_wall_out: float  # Dispersione muri -> esterno (kW/°C) [trasmittanza involucro]
  h_air_wall: float  # Accoppiamento convettivo/radiante interno aria <-> muri (kW/°C)
  max_heater_kw: float = 10.0  # Potenza max caldaia/pompa di calore (kW)


# Profili tarati per replicare esattamente i dati del video (Outside: 5°C, Target: 21°C)
HOUSE_PRESETS = {
    "standard": HouseProfile(
        name="Standard House",
        c_air=0.40,
        c_wall=27.0,
        h_air_out=0.088,
        h_wall_out=0.170,
        h_air_wall=0.72,
        max_heater_kw=10.0,
    ),
    "drafty": HouseProfile(
        name="Drafty Old",
        c_air=0.45,
        c_wall=22.0,
        h_air_out=0.230,
        h_wall_out=0.280,
        h_air_wall=0.85,
        max_heater_kw=12.0,
    ),
    "insulated": HouseProfile(
        name="Insulated",
        c_air=0.38,
        c_wall=32.0,
        h_air_out=0.025,
        h_wall_out=0.065,
        h_air_wall=0.65,
        max_heater_kw=8.0,
    ),
}


class HeatingSimulation:

  def __init__(
      self,
      house: HouseProfile,
      t_outside: float = 5.0,
      t_target: float = 21.0,
      away_start_hour: float = 8.0,
      away_end_hour: float = 17.0,
      reheat_lead_minutes: int = 60,  # Accensione anticipata (16:00 per essere caldi alle 17:00)
  ):
    self.house = house
    self.t_outside = t_outside
    self.t_target = t_target
    self.away_start_min = int(away_start_hour * 60)
    self.away_end_min = int(away_end_hour * 60)
    self.reheat_start_min = self.away_end_min - reheat_lead_minutes
    self.dt_hours = 1.0 / 60.0  # Passo temporale di 1 minuto

  def _steady_state_temperatures(self) -> tuple[float, float]:
    """Calcola le temperature stazionarie iniziali di aria e muri a regime con riscaldamento attivo."""
    t_air = self.t_target
    # In regime stazionario: h_air_wall * (t_air - t_wall) = h_wall_out * (t_wall - t_out)
    t_wall = (
        self.house.h_air_wall * t_air + self.house.h_wall_out * self.t_outside
    ) / (self.house.h_air_wall + self.house.h_wall_out)
    return t_air, t_wall

  def run(self, strategy: str = "always_on") -> dict:
    """Esegue la simulazione sulle 24 ore (1440 minuti)."""
    t_air, t_wall = self._steady_state_temperatures()

    minutes = 24 * 60
    history = {
        "time_min": np.arange(minutes),
        "t_air": np.zeros(minutes),
        "t_wall": np.zeros(minutes),
        "heater_kw": np.zeros(minutes),
        "leak_kw": np.zeros(minutes),
        "energy_kwh": np.zeros(minutes),
    }

    cum_energy = 0.0

    for m in range(minutes):
      # Dispersione termica istantanea verso l'esterno
      leak_air = self.house.h_air_out * (t_air - self.t_outside)
      leak_wall = self.house.h_wall_out * (t_wall - self.t_outside)
      total_leak = leak_air + leak_wall

      # Controllo logico del termostato
      if strategy == "always_on":
        heater_enabled = True
      elif strategy == "off_while_out":
        # Spento tra l'uscita e l'ora di avvio del pre-riscaldamento
        if self.away_start_min <= m < self.reheat_start_min:
          heater_enabled = False
        else:
          heater_enabled = True
      else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")

      # Modulazione del riscaldatore con banda proporzionale per evitare oscillazioni
      if heater_enabled:
        temp_error = self.t_target - t_air
        if temp_error <= 0.0:
          heater_kw = 0.0
        else:
          # Guadagno proporzionale per ripristino rapido fino al limite max
          p_req = temp_error * 15.0
          heater_kw = min(self.house.max_heater_kw, max(0.0, p_req))
      else:
        heater_kw = 0.0

      # Flusso di calore aria <-> parete interna
      q_air_to_wall = self.house.h_air_wall * (t_air - t_wall)

      # Integrazione di Eulero esplicito al minuto
      d_t_air = (
          (heater_kw - leak_air - q_air_to_wall)
          / self.house.c_air
          * self.dt_hours
      )
      d_t_wall = (
          (q_air_to_wall - leak_wall) / self.house.c_wall * self.dt_hours
      )

      t_air += d_t_air
      t_wall += d_t_wall
      cum_energy += heater_kw * self.dt_hours

      history["t_air"][m] = t_air
      history["t_wall"][m] = t_wall
      history["heater_kw"][m] = heater_kw
      history["leak_kw"][m] = total_leak
      history["energy_kwh"][m] = cum_energy

    return history


def print_comparison_table(results: dict):
  """Stampa un resoconto testuale sintetico con percentuali di risparmio."""
  print("\n" + "=" * 78)
  print(
      f"{'Profilo Casa':<18} | {'Always On (kWh)':<16} | {'Off & Reheat (kWh)':<18} | {'Risparmio':<10} | {'T min':<6}"
  )
  print("-" * 78)

  for data in results.values():
    on_kwh = data["always_on"]["energy_kwh"][-1]
    off_kwh = data["off_while_out"]["energy_kwh"][-1]
    saving_pct = (on_kwh - off_kwh) / on_kwh * 100.0
    min_temp = np.min(data["off_while_out"]["t_air"])

    print(
        f"{data['name']:<18} | {on_kwh:14.1f}   | {off_kwh:16.1f}   | {saving_pct:8.1f}% | {min_temp:4.1f}°C"
    )
  print("=" * 78 + "\n")


def plot_simulation(sim_std: HeatingSimulation, res_on: dict, res_off: dict):
  """Traccia i grafici comparativi su 24 ore identici al layout del video."""
  hours = res_on["time_min"] / 60.0

  fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

  # 1. Temperature
  ax = axes[0]
  ax.plot(
      hours,
      res_on["t_air"],
      label="T Aria (Always On)",
      color="#e67e22",
      linewidth=2,
  )
  ax.plot(
      hours,
      res_off["t_air"],
      label="T Aria (Off While Out)",
      color="#2ecc71",
      linewidth=2,
  )
  ax.plot(
      hours,
      res_off["t_wall"],
      label="T Pareti (Off While Out)",
      color="#3498db",
      linestyle="--",
      linewidth=1.5,
  )
  ax.axhline(
      sim_std.t_outside,
      color="#95a5a6",
      linestyle=":",
      label=f"Esterno ({sim_std.t_outside}°C)",
  )
  ax.axvspan(
      sim_std.away_start_min / 60.0,
      sim_std.away_end_min / 60.0,
      color="#ecf0f1",
      alpha=0.6,
      label="Fuori casa (08-17)",
  )
  ax.set_ylabel("Temperatura (°C)")
  ax.set_title(
      f"Simulazione Termica Casa Standard (Esterno: {sim_std.t_outside}°C, Target: {sim_std.t_target}°C)"
  )
  ax.grid(True, alpha=0.3)
  ax.legend(loc="lower right", fontsize=8)

  # 2. Potenza Riscaldatore (kW)
  ax = axes[1]
  ax.plot(
      hours,
      res_on["heater_kw"],
      label="Heater Always On",
      color="#e67e22",
      alpha=0.7,
  )
  ax.plot(
      hours,
      res_off["heater_kw"],
      label="Heater Off While Out",
      color="#2ecc71",
      linewidth=2,
  )
  ax.set_ylabel("Potenza (kW)")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="upper right", fontsize=8)

  # 3. Energia Cumulativa (kWh)
  ax = axes[2]
  ax.plot(
      hours,
      res_on["energy_kwh"],
      label=f"Always On ({res_on['energy_kwh'][-1]:.1f} kWh)",
      color="#e67e22",
      linewidth=2,
  )
  ax.plot(
      hours,
      res_off["energy_kwh"],
      label=f"Off While Out ({res_off['energy_kwh'][-1]:.1f} kWh)",
      color="#2ecc71",
      linewidth=2,
  )
  diff = res_on["energy_kwh"][-1] - res_off["energy_kwh"][-1]
  pct = diff / res_on["energy_kwh"][-1] * 100.0
  ax.set_ylabel("Energia (kWh)")
  ax.set_xlabel("Ora del giorno (08:00 -> 08:00 giorno successivo)")
  ax.set_title(f"Risparmio netto a fine giornata: {diff:.1f} kWh (-{pct:.1f}%)")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="upper left", fontsize=8)

  plt.tight_layout()
  return fig


if __name__ == "__main__":
  # ==============================================================
  # PARAMETRI PERSONALIZZABILI
  # ==============================================================
  T_OUTSIDE = 5.0  # Prova con -5.0, 0.0, 10.0 °C
  T_TARGET = 21.0  # Temperatura interna desiderata
  AWAY_START = 8.0  # Uscita: 08:00
  AWAY_END = 17.0  # Rientro: 17:00
  PREHEAT_MINUTES = (
      60  # Accensione alle 16:00 (60 min prima) per arrivare caldi a casa
  )

  results = {}

  for key, house in HOUSE_PRESETS.items():
    sim = HeatingSimulation(
        house=house,
        t_outside=T_OUTSIDE,
        t_target=T_TARGET,
        away_start_hour=AWAY_START,
        away_end_hour=AWAY_END,
        reheat_lead_minutes=PREHEAT_MINUTES,
    )
    res_on = sim.run(strategy="always_on")
    res_off = sim.run(strategy="off_while_out")
    results[key] = {
        "name": house.name,
        "always_on": res_on,
        "off_while_out": res_off,
    }

  # Stampa tabella comparativa per i 3 profili
  print_comparison_table(results)

  # Grafico dettagliato per il profilo standard
  sim_std = HeatingSimulation(
      house=HOUSE_PRESETS["standard"],
      t_outside=T_OUTSIDE,
      t_target=T_TARGET,
      away_start_hour=AWAY_START,
      away_end_hour=AWAY_END,
      reheat_lead_minutes=PREHEAT_MINUTES,
  )
  fig = plot_simulation(
      sim_std,
      results["standard"]["always_on"],
      results["standard"]["off_while_out"],
  )
  fig.savefig("simulazione_riscaldamento.png", dpi=120)
  plt.show()
