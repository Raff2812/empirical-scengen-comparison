## Arricchimento dei log

Dopo la raccolta, i log grezzi devono essere arricchiti con le metriche.
Questo produce JSON completi pronti per l’analisi.

### Esecuzione

```bash
cd data_gathering/enriching
python orchestrator.py --input_dir <cartella_input> --output_dir <cartella_output>
```

Opzioni:

* `--input_dir` : directory con i file json prodotti dal logger `*_log_basic.json`
* `--output_dir` :  directory di destinazione dei log arricchiti

### Output

Ogni file arricchito avrà una sezione aggiuntiva in `results` con:

* `critical_metrics`: MDBV, minTTC, TET
* `functional_metrics`: completion rate, route stability, tempo di completamento
* `dynamics_metrics` : velocità e accelerazioni

## Esempio di log arricchito

```json
{
  "tool": "SimADFuzz",
  "scenario_id": "sc-42",
  "generation_id": "gen-1",
  "results": {
    "has_collision": true,
    "has_speeding": false,
    "critical_metrics": {
      "MDBV": 1.42,
      "MDBV_frame": 123,
      "MDBV_actor": { "id": "37", "type_id": "vehicle.audi.tt" },
      "min_TTC": 0.27,
      "min_TTC_frame": 118,
      "min_TTC_actor": { "id": "37", "type_id": "vehicle.audi.tt" },
      "TET_total": 6.35,
      "TET_max": 2.45,
      "TET_max_start_frame": 100,
      "TET_max_end_frame": 149,
      "MDBV_per_actor": [
        { "actor_id": "37", "type_id": "vehicle.audi.tt", "min_distance": 1.42, "frame": 123 }
      ]
    },
    "functional_metrics": {
      "completion_rate": 95.3,
      "route_following_stability": 87.5,
      "time_to_completion": 62.1,
      "total_planned_distance": 134.2,
      "actual_distance_traveled": 128.7,
      "max_progress_reached": 133.9,
      "deviation_stats": {
        "mean": 0.82,
        "rmse": 1.03,
        "mae": 0.71,
        "max_deviation": 2.35,
        "std_dev": 0.44
      },
      "completion_frame": 1230,
      "completion_timestamp": 61.9
    },
    "dynamics_metrics": {
      "mean_speed": 8.9,
      "max_speed": 14.3,
      "mean_long_acc": 0.82,
      "p95_long_acc": 2.35,
      "max_long_acc": 4.1
    }
  }
}
```
---
Una volta arricchiti i file, fare riferimento alla [guida all'analisi](pipeline.md)