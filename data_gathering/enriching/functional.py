from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


@dataclass(frozen=True)
class DeviationStats:
    """Contiene le statistiche sulle deviazioni laterali dalla rotta."""
    mean: float
    rmse: float
    mae: float
    max_deviation: float
    std_dev: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__

@dataclass(frozen=True)
class EgoJourney:
    """Rappresenta il percorso effettivo del veicolo ego"""
    positions: np.ndarray
    timestamps: np.ndarray
    frame_ids: np.ndarray

@dataclass(frozen=True)
class FunctionalMetricResult:    
    """Risultato finale e aggregato delle metriche funzionali."""
    completion_rate: float
    route_following_stability: float
    time_to_completion: Optional[float]
    total_planned_distance: float
    actual_distance_traveled: float
    max_progress_reached: float
    deviation_stats: DeviationStats
    completion_frame: Optional[int]
    completion_timestamp: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completion_rate": self.completion_rate,
            "route_following_stability": self.route_following_stability,
            "time_to_completion": self.time_to_completion,
            "total_planned_distance": self.total_planned_distance,
            "actual_distance_traveled": self.actual_distance_traveled,
            "max_progress_reached": self.max_progress_reached,
            "deviation_stats": self.deviation_stats.to_dict(),
            "completion_frame": self.completion_frame,
            "completion_timestamp": self.completion_timestamp,
        }


class Route:
    """
    Rappresenta una rotta pianificata come una spezzata 2D (polilinea).
    È responsabile di tutta la logica geometrica relativa alla rotta stessa.
    """
    def __init__(self, waypoints: List[List[float]]):
        if len(waypoints) < 2:
            raise ValueError("Una rotta valida richiede almeno 2 waypoints")
        
        #prendi solo coordinate x, y
        waypoints_2d = np.array(waypoints)[:, :2] 

        #vettori che rappresentano la "differenza" tra waypoints consecutivi
        segment_vectors = np.diff(waypoints_2d, axis=0) 

        #calcolo del modulo di questi vettori
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        valid_segment_mask = segment_lengths > 1e-9
        if not np.any(valid_segment_mask):
            raise ValueError("La rotta non contiene segmenti di lunghezza valida")
        
        # aggiungiamo true in testa per includere sempre il primo waypoint.
        # serve ad allineare la maschera ai waypoint (non ai segmenti).
        waypoints_mask = np.insert(valid_segment_mask, 0, True)

        valid_indices = np.where(waypoints_mask)[0]
        self.waypoints = waypoints_2d[valid_indices]
        self.segment_lengths = segment_lengths[valid_segment_mask]

        # Somma cumulativa delle lunghezze dei segmenti
        segment_cumsum = np.cumsum(self.segment_lengths)

        # Inseriamo 0 all'inizio per indicare che la distanza iniziale è zero
        self.cumulative_distances = np.insert(segment_cumsum, 0, 0.0)

        # La lunghezza totale della rotta è l'ultima distanza cumulata
        self.total_length = self.cumulative_distances[-1]

    def project(self, point: np.ndarray):
        """
        Proietta un punto 2D sulla rotta con lo scopo di trovare
        il punto sulla linea della rotta che è più vicino al punto dato.

        Returns:
            - s (float): Progresso curvilineo (distanza dall'inizio della rotta al punto proiettato).
            - d (float): Deviazione laterale (distanza tra il punto e la sua proiezione).
        """
        p = point[:2]
        min_lateral_dist = float("inf") # deviazione laterale più piccola trovata finora
        best_progress = 0.0  # progresso curvilineo corrispondente

        for i in range(len(self.segment_lengths)):
            a = self.waypoints[i]   #punto iniziale segmento
            b = self.waypoints[i+1] #punto finale segmento

            #proiezione del punto p sul segmento [a, b]
            t, lateral_dist = self.project_on_segment(p, a, b)

            # se questa proiezione è più vicina alla rotta, la salviamo
            if lateral_dist < min_lateral_dist:
                min_lateral_dist = lateral_dist

                # calcola il progresso lungo questo segmento
                progress_on_segment = t * self.segment_lengths[i]

                # progresso totale = distanza cumulativa + progresso nel segmento
                best_progress = self.cumulative_distances[i] + progress_on_segment

        return best_progress, min_lateral_dist
    
    @staticmethod
    def project_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
        """
        Proietta un punto 'p' su un singolo segmento 'ab'.
        """
        # vettore dal punto 'a' al punto 'b'
        ab = b - a
        # vettore dal punto 'a' al punto 'p'
        ap = p - a

        ab_len_sq = np.dot(ab, ab) # lunghezza al quadrato del segmento 'ab'

        #se la lunghezza del segmento è praticamente 0, esso è paragonabile ad un punto
        if ab_len_sq < 1e-9:
            return 0.0, np.linalg.norm(ap) 
        
        #prodotto scalare tra ab e ap
        dot_product = np.dot(ap, ab)
   
        # Dividendo per la lunghezza al quadrato di 'ab', normalizziamo questo valore
        # per ottenere 't', un parametro che ci dice dove cade la proiezione.
        # - Se t = 0, la proiezione è su 'a'.
        # - Se t = 1, la proiezione è su 'b'.
        # - Se 0 < t < 1, la proiezione è tra 'a' e 'b'.
        # - Se t < 0 o t > 1, la proiezione è al di fuori del segmento 'ab'.
        # volendo ottenere la proiezione sul segmento ab, blocchiamo t nell'intervallo [0, 1]
        t = np.clip((dot_product / ab_len_sq), 0.0, 1.0)

        # calcolo delle coordinate del punto proiettato
        # si parte da 'a' e ci si sposta lungo la direzione del segmento 'ab'
        # di un fattore 't'
        projection_point = a + ab * t

        # calcolo della distanza laterale è uguale alla distanza euclidea tra
        # il punto originale 'p' e la sua ombra 'projection_point'
        distance = np.linalg.norm(p - projection_point)

        return t, distance
    
def compute_deviation_stats(deviations: np.ndarray) -> DeviationStats:
    """Calcola le statistiche di base da un array di deviazioni laterali."""
    if deviations.size == 0:
        return DeviationStats(0, 0, 0, 0, 0)
    return DeviationStats(
        mean=float(np.mean(deviations)),
        rmse=float(np.sqrt(np.mean(deviations**2))),
        mae=float(np.mean(np.abs(deviations))),
        max_deviation=float(np.max(deviations)),
        std_dev=float(np.std(deviations))
    )

def compute_route_following_stability(mean_devation:float, threshold: float) -> float:
    """Calcola un punteggio di stabilità (0-100) basato sulla deviazione media."""
    # Se la deviazione è nulla: stabilità perfetta
    if mean_devation <= 0:
        return 100.0
    
    # Calcola un coefficiente negativo proporzionale alla deviazione
    # all'aumentare della deviazione il coefficiente (decay) diventa ancora più piccolo
    decay = -mean_devation / threshold

    # Converte la deviazione in punteggio tramite decadimento esponenziale
    score = 100 * np.exp(decay)

    return float(np.clip(score, 0.0, 100.0))

def compute_traveled_distance(positions: np.ndarray) -> float:
    """Calcola la distanza totale percorsa dal veicolo."""
    if len(positions) < 2:
        return 0.0
    
    # Differenze tra posizioni consecutive (solo coordinate x,y)
    # otteniamo i vettori spostamento tra ogni coppia di punti
    displacements = np.diff(positions[:, :2], axis=0)

    # Calcoliamo la lunghezza (norma) di ogni spostamento
    # queste sono le distanze percorse nei singoli segmenti
    segment_lengths = np.linalg.norm(displacements, axis=1)

    # Sommiamo tutte le distanze per ottenere il percorso totale
    return float(np.sum(segment_lengths))

def find_completion_time(journey: EgoJourney, progress: np.ndarray, total_dist: float, tolerance: float) -> Tuple:
    """Identifica il primo istante in cui la missione è considerata completata."""
    goal_dist = total_dist - tolerance
    completed_indices = np.where(progress >= goal_dist)[0]

    if completed_indices.size > 0:
        idx = completed_indices[0]
        return int(journey.frame_ids[idx]), float(journey.timestamps[idx]), journey.timestamps[idx] - journey.timestamps[0]
    
    return None, None, None


class FunctionalAnalyzer:
    """
    Orchestra il processo di calcolo delle metriche di performance.
    """
    def __init__(self, output_dir, completion_tolerance: float = 10.0, stability_threshold: float = 5.0):
        self.completion_tolerance = completion_tolerance
        self.stability_threshold = stability_threshold
        self.output_dir = output_dir

    def analyze(self, log_data: Dict[str, Any]) -> FunctionalMetricResult:
        """Esegue l'analisi completa dei log e restituisce le metriche."""
        
        #Estrazione dei dati e salvataggio nelle strutture dati definite precedentemente
        route = Route(log_data.get("mission", {}).get("waypoints", []))
        journey = self.extract_journey_from_log(log_data.get("frames", []))

        if journey.positions.shape[0] == 0:
            projections = np.empty((0, 2))
        else:
            projections = np.array([route.project(pos) for pos in journey.positions])

        progress = projections[:, 0] if projections.size > 0 else np.array([])
        deviations = projections[:, 1] if projections.size > 0 else np.array([])

        #Calcolo metriche
        max_progress = float(np.max(progress)) if progress.size > 0 else 0.0
        completion_rate = (max_progress / route.total_length) * 100 if route.total_length > 0 else 0.0
        
        completion_frame, completion_ts, ttc = find_completion_time(
            journey, progress, route.total_length, self.completion_tolerance
        )

        deviation_stats = compute_deviation_stats(deviations)
        stability_score = compute_route_following_stability(deviation_stats.mean, self.stability_threshold)
        actual_distance = compute_traveled_distance(journey.positions)

        return FunctionalMetricResult(
            completion_rate, stability_score, ttc, route.total_length,
            actual_distance, max_progress, deviation_stats, completion_frame, completion_ts
        )


    def extract_journey_from_log(self, frames: List[Dict]) -> EgoJourney:
        """Estrae e pulisce i dati del percorso dell'ego dai log."""
        positions, timestamps, frame_ids = [], [], []

        for f in frames:
            loc = f.get("ego_vehicle", {}).get("location")
            if loc:
                positions.append(loc[:3])
                timestamps.append(f.get("timestamp", 0.0))
                frame_ids.append(f.get("frame", len(frame_ids)))

        return EgoJourney(np.array(positions), np.array(timestamps), np.array(frame_ids))
    
    def analyze_to_dict(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Metodo per ottenere il risultato direttamente come dizionario."""
        result = self.analyze(log_data)
        return {"performance": result.to_dict()}

