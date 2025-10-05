## Struttura di un log generato

Ogni simulazione produce un file JSON con la seguente struttura (esempio semplificato):

```json
{
  "tool": "SimADFuzz",
  "generation_id": "gen-1",
  "scenario_id": "sc-42",
  "map_name": "Town03",
  "start_time": "2025-09-30_15-22-01",
  "simulation_start_time": 1738290101.23,
  "delta_time": 0.05,

  "results": {
    "has_collision": true,
    "has_red_light_violation": false,
    "has_speeding": true,
    "has_stop_violation": false,
    "has_stuck": false,
    "event_counts": {
      "collision": 1,
      "red_light": 0,
      "speeding": 1,
      "stop_sign": 0,
      "stuck": 0
    },
    "total_frames": 1240,
    "total_simulation_time": 62.1,
    "simulated_time": 61.9
  },

  "events": {
    "collision": [...],
    "red_lights": [...],
    "speeding": [...],
    "stuck": [...],
    "stop_sign": [...]
  },

  "mission": {
    "start_location": [...],
    "end_location": [...],
    "waypoints": [...]
  },

  "actors": {
    "1": { "type_id": "vehicle.tesla.model3", "role": "ego", ... },
    "37": { "type_id": "vehicle.audi.tt", "role": "npc", ... }
  },

  "frames": [
    { "frame": 0, "ego_vehicle": {...}, "other_actors": {...} }
  ]
}
```
---
