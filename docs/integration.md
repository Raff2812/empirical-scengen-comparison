## Integrazione del sistema di logging

Per generare log compatibili con la pipeline è necessario integrare il logger negli script esecutivi di CARLA.
L’integrazione del framework di raccolta dati richiede solo di copiare la cartella `data_gathering` nella sorgente del tool analizzato.
Una volta copiato il modulo, è sufficiente seguire i passaggi riportati di seguito.

1. **Import dei moduli necessari nello script che si occupa di gestire la creazione degli scenari**

   ```python
   from data_gathering.carlaBasicLogger import CarlaBasicLogger, LOGGER_REGISTRY
   from data_gathering.violationMonitor import ViolationMonitor
   ```

2. **Istanziazione del logger con i relativi parametri**

   ```python
   logger = CarlaBasicLogger(
       tool="NomeTool",
       generation_id=generation_id,
       scenario_id=scenario_id,
       output_dir="temp_dir/json",
       world=world,
       client=client,
       record_binary=True
   )
   ```
> Nota: qui si parla di generation_id e scenario_id in quanto i tool analizzati erano tutti algoritmi genetici.


3. **Registrazione del veicolo ego e della sua missione, subito dopo lo spawn**

   ```python
   logger.register_ego_actor(ego_vehicle=ego, snapshot=world.get_snapshot())
   logger.set_mission_from_agent(ego_agent=ego_agent, ego_sp=ego_sp, ego_dp=ego_dp)
   ```
> Nota: l'ego agent considerato è il BehaviorAgent. Per altri tipi di agenti bisognerà modificare il codice.

4. **Setup del violation monitor, legato al logger precedentemente istanziato**

   ```python
   LOGGER_REGISTRY[ego.id] = logger
   violation_monitor = ViolationMonitor(ego_vehicle=ego, logger=logger)
   logger.violation_monitor = violation_monitor
   ```

5. **Gestione delle collisioni affidata al logger (metodo statico)**

   ```python
   carla_collision_sensor.listen(
       lambda event: CarlaBasicLogger.handle_collision(event, state)
   )
   ```

6. **Update frame-by-frame nel ciclo che gestisce la simulazione dello scenario**

   ```python
   snapshot = world.get_snapshot()
   logger.update_frame(world=world, ego_vehicle=ego, snapshot=snapshot)
   ```

7. **Finalizzazione e salvataggio dei risultati, fuori dal ciclo**

   ```python
   logger.finalize_and_save()
   ```

**Linee guida generali**
* L’istanza del logger subito dopo la connessione a CARLA.
* `logger.update_frame` in ogni ciclo di simulazione.
* `finalize_and_save` sempre a fine simulazione (anche in caso di early stop).
---

## Esempio di integrazione
  ```python
import carla
from PythonAPI.carla.agents.navigation.custom_behavior_agent import CustomBehaviorAgent

from data_gathering.carlaBasicLogger import CarlaBasicLogger, LOGGER_REGISTRY
from data_gathering.violationMonitor import ViolationMonitor

# Connessione a CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()

# Spawn ego vehicle
blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
spawn_point = world.get_map().get_spawn_points()[0]
ego = world.spawn_actor(blueprint, spawn_point)

# Configura BehaviorAgent (in questo caso custom con gestione stop)
ego_agent = CustomBehaviorAgent(ego, behavior="cusotm")
ego_agent.set_destination(world.get_map().get_spawn_points()[1].location)

# === Integrazione logger ===
logger = CarlaBasicLogger(
    tool="ExampleTool",
    generation_id="gen-0",
    scenario_id="sc-0",
    output_dir="logs/json",
    world=world,
    client=client,
    record_binary=True
)
LOGGER_REGISTRY[ego.id] = logger

logger.register_ego_actor(ego_vehicle=ego, snapshot=world.get_snapshot())
logger.set_mission_from_agent(ego_agent, ego_agent._local_planner._start_waypoint, ego_agent._destination)

violation_monitor = ViolationMonitor(ego_vehicle=ego, logger=logger)
logger.violation_monitor = violation_monitor

collision_sensor = world.spawn_actor(
    world.get_blueprint_library().find("sensor.other.collision"),
    carla.Transform(),
    attach_to=ego
)
collision_sensor.listen(lambda event: CarlaBasicLogger.handle_collision(event, {}))

# Ciclo di simulazione
for _ in range(500):
    world.tick()
    snapshot = world.get_snapshot()
    logger.update_frame(world=world, ego_vehicle=ego, snapshot=snapshot)
    #tutta la logica di generazione
    

#salvataggio del file JSON
logger.finalize_and_save()
print("Scenario terminato, log salvato.")
```
Il file JSON prodotto ha tale [struttura](file_structure.md)

Dunque, integrando in questa maniera nell'**output_dir** verrà salvato il relativo file JSON.
Questo può e deve essere arricchito per effettuare l'analisi tramite i moduli presenti in **data_gathering/enriching**, vedi [enrichment](enrichment.md).
