from typing import TYPE_CHECKING, Dict, List

import carla

if TYPE_CHECKING:
    from carlaBasicLogger import CarlaBasicLogger

from utils.carla_help import get_speed_mps, get_speed_kmh

class ViolationMonitor:
    """
    Monitora in tempo reale alcune violazioni comportamentali del veicolo ego durante la simulazione.
    Registra direttamente gli eventi nel logger associato.

    Supporta attualmente:
    - attraversamento con semaforo rosso
    - superamento del limite di velocitò
    - passaggio con il segnale di stop (behavior agent esteso per garantire una rilevazione euristica)
    """

    def __init__(self, ego_vehicle: carla.Vehicle, logger: "CarlaBasicLogger"):
        """
        Inizializza il sistema di monitoraggio violazioni.

        Args:
            ego_vehicle: veicolo controllato (ego)
            logger: istanza del CarlaBasicLogger associato
        """
        self.ego_vehicle = ego_vehicle
        self.logger = logger
        self.world = self.logger.world
        self.map = self.world.get_map()
        self.fps = int(1 / self.world.get_settings().fixed_delta_seconds)

        #######################################################################
        #Parametri di configurazione 
        #######################################################################
        #Parametri universali
        self.min_speed_threshold_mps = 0.3 #soglia minima di velocità (mps) per considerare veicolo fermo
        #######################################################################
        #Gestione speeding
        self.last_speed_limit = None #ultimo limite di velocità rilevato (km/h)
        self.last_speeding_frame = -1 #frame in cui è stata rilevata l'ultima violazione di velocità
        self.frame_speed_lim_changed = -1 #frame in cui è cambiato il limite di velocità
        self.speed_grace_T = 3  #durata (in secondi) del periodo di tolleranza dopo un cambio limite
        self.speed_margin_kmh = 5.0 #margine di tolleranza oltre il limite (km/h) prima di considerare infrazione
        ########################################################################
        #Gestione red light
        self.on_red = False
        self.on_red_speeds = []
        self.red_light_violations_timestamps: Dict[int, float] = {} #insieme degli id dei semafori già violati (per evitare log_duplicati)
        self.red_light_cooldown_s = 10.0  # tempo minimo (in secondi) per riconsiderare lo stesso semaforo
        ########################################################################
        # Gestione degli stops signs
        self.stop_signs: List[carla.TrafficSign] = self.world.get_actors().filter("*stop*")
        self.stop_proximity_threshold = 15.0  # m
        self.stop_required_duration = 2.0  # s
        self.stop_cooldown_s = 5.0  # tempo minimo tra due violazioni sullo stesso stop
        
        # Tracking dello stato stop per ogni stop sign
        self.stop_states = {}  # {stop_id: {"entered_time": float, "stopped_time": float, "was_stopped": bool}}
        self.current_stop_violations = set()  # ID degli stop già violati questo frame


    def tick(self, frame:int, timestamp:float):
        """
        Chiamato ad ogni frame della simulazione. Controlla tutte le violazioni ad ora abilitate
        """
        self.check_speeding(frame, timestamp)
        self.check_red_light(frame, timestamp)
        self.check_stop_signs(frame, timestamp)

    def check_speeding(self, frame:int, timestamp:float):
        """
        Verifica se il veicolo ha superato il limite di velocità corrente.
        """
        speed_kmh = get_speed_kmh(self.ego_vehicle)

        speed_limit_kmh = self.ego_vehicle.get_speed_limit()

        # 1. cambio limite: aggiorna
        if self.last_speed_limit != speed_limit_kmh:
            self.frame_speed_lim_changed = frame
            self.last_speed_limit = speed_limit_kmh

        # 2. grace period dopo cambio limite
        grace_frames = int(self.speed_grace_T * self.fps)
        if frame < self.frame_speed_lim_changed + grace_frames:
            return  # non controllare ancora

        # 3. verifica speeding
        if speed_kmh > speed_limit_kmh + self.speed_margin_kmh:
            if frame != self.last_speeding_frame:
                self.logger.log_speeding(frame, timestamp, speed_kmh, speed_limit_kmh)
                self.last_speeding_frame = frame

    def check_red_light(self, frame: int, timestamp: float):
        lights = self.ego_vehicle.get_traffic_light()
        at_traffic_light = self.ego_vehicle.is_at_traffic_light()
        
        if lights is not None and lights.state == carla.TrafficLightState.Red and at_traffic_light:
            # Registra che siamo al semaforo rosso
            speed = get_speed_kmh(self.ego_vehicle)
            if not self.on_red:
                self.on_red = True
                self.on_red_speeds = []
            self.on_red_speeds.append(speed)
        
        # Controlla violazione quando usciamo dal trigger box, indipendentemente dal colore
        elif not at_traffic_light and self.on_red:
            # Uscita dal trigger box - valuta violazione
            self.on_red = False
            if all(s > 0.1 for s in self.on_red_speeds):
                # Violazione rilevata
                last_violation_time = self.red_light_violations_timestamps.get(lights.id if lights else -1, -999.0)
                if timestamp - last_violation_time > self.red_light_cooldown_s:
                    self.logger.log_red_light(frame, timestamp, max(self.on_red_speeds))
                    self.red_light_violations_timestamps[lights.id if lights else -1] = timestamp

    def check_stop_signs(self, frame: int, timestamp: float):
        """
        Verifica violazioni legate agli stop sign.
        """
        ego_location = self.ego_vehicle.get_location()
        ego_speed_mps = get_speed_mps(self.ego_vehicle)
        is_ego_stopped = ego_speed_mps <= self.min_speed_threshold_mps

        # 1. Identifica gli stop attualmente "attivi" (quelli in cui si trova l'ego)
        active_stop_ids = set()
        for stop_sign in self.stop_signs:
            if ego_location.distance(stop_sign.get_location()) < self.stop_proximity_threshold:
                trigger_volume: carla.BoundingBox = stop_sign.trigger_volume
                if trigger_volume.contains(ego_location, stop_sign.get_transform()):
                    active_stop_ids.add(stop_sign.id)

        # 2. Giudica gli stop che non sono più attivi (il veicolo è appena uscito)
        stops_to_remove = []
        for stop_id, state in self.stop_states.items():
            if stop_id not in active_stop_ids:
                # Il veicolo è uscito dal trigger, è il momento di giudicare
                if not state["was_stopped"]:
                    # Violazione: non si è fermato correttamente
                    last_time = state.get("last_violation_time", -999.0)
                    if timestamp - last_time > self.stop_cooldown_s:
                        self.logger.log_stop_violation(
                            frame=frame,
                            timestamp=timestamp,
                            lm_id=stop_id,
                            lm_loc=state["location"],
                            speed_kmh=get_speed_kmh(self.ego_vehicle),
                            stopped=False
                        )
                        # Anche se non c'è più lo stato, memorizziamo l'ultima violazione
                        # per gestire il cooldown nel caso in cui si rientri subito.
                        self.stop_states[stop_id]['last_violation_time'] = timestamp

                # Aggiungi alla lista per la pulizia
                stops_to_remove.append(stop_id)

        # Pulisce gli stati degli incontri terminati
        for stop_id in stops_to_remove:
            if stop_id in self.stop_states:
                del self.stop_states[stop_id]

        # 3. Aggiorna o crea lo stato per gli stop attualmente attivi
        for stop_id in active_stop_ids:
            # Se è un nuovo incontro, crea lo stato
            if stop_id not in self.stop_states:
                stop_actor = self.world.get_actor(stop_id)
                if stop_actor:
                    self.stop_states[stop_id] = {
                        "stopped_time": None,
                        "was_stopped": False,
                        "location": stop_actor.get_location()
                    }

            # Aggiorna lo stato corrente
            if stop_id in self.stop_states:
                state = self.stop_states[stop_id]
                
                # Se l'ego si ferma per la prima volta durante questo incontro
                if is_ego_stopped and state["stopped_time"] is None:
                    state["stopped_time"] = timestamp
                
                # Se l'ego non è fermo, resetta il timer di sosta
                if not is_ego_stopped:
                    state["stopped_time"] = None
                
                # Se è stato fermato abbastanza a lungo, marca l'incontro come "conforme"
                if state["stopped_time"] is not None:
                    stop_duration = timestamp - state["stopped_time"]
                    if stop_duration >= self.stop_required_duration:
                        state["was_stopped"] = True