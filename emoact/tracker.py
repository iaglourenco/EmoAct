"""
Face tracking module for maintaining consistent person IDs across video frames.
Uses face embeddings and spatial proximity to track individuals.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from emoact.types import FrameInfo, PersonInfo
from emoact.utils import bbox_distance, cosine_similarity


class FaceTracker:
    """
    Tracks faces across video frames using embeddings and spatial information.

    Attributes:
        similarity_threshold: Minimum cosine similarity for face matching (0-1)
        max_distance_threshold: Maximum pixel distance for considering same person
        max_frames_missing: Maximum frames a person can be missing before ID is retired
        next_person_id: Counter for generating new person IDs
        tracked_persons: Dictionary storing information about tracked persons
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        max_distance_threshold: float = 200.0,
        max_frames_missing: int = 30,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_distance_threshold = max_distance_threshold
        self.max_frames_missing = max_frames_missing
        self.next_person_id = 1
        self.tracked_persons: Dict[str, Dict] = {}
        self.archived_tracks = {}  # Armazena tracks finalizados

    def reset(self):
        """Reset tracker state."""
        self.next_person_id = 1
        self.tracked_persons = {}

    def _calculate_match_score(
        self,
        face_embedding: np.ndarray,
        face_location: Tuple,
        tracked_embedding: np.ndarray,
        tracked_location: Tuple,
        embedding_weight: float = 0.7,
    ) -> float:
        """
        Calculate overall match score combining embedding similarity and spatial proximity.

        """
        # Calculate embedding similarity
        emb_similarity = cosine_similarity(face_embedding, tracked_embedding)

        # Calculate spatial proximity score
        distance = bbox_distance(face_location, tracked_location)
        spatial_score = max(0.0, 1.0 - (distance / self.max_distance_threshold))

        # Combine scores with weighted average
        match_score = (
            embedding_weight * emb_similarity + (1 - embedding_weight) * spatial_score
        )

        return match_score

    def _find_best_match(
        self, face_embedding: np.ndarray, face_location: Tuple, frame_number: int
    ) -> Optional[str]:
        """
        Find the best matching tracked person for a detected face.

        """
        best_person_id = None
        best_score = 0.0

        for person_id, person_data in self.tracked_persons.items():
            # Skip persons that have been missing for too long
            frames_missing = frame_number - person_data["last_seen_frame"]
            if frames_missing > self.max_frames_missing:
                continue

            # Calculate match score
            score = self._calculate_match_score(
                face_embedding,
                face_location,
                person_data["embedding"],
                person_data["last_location"],
            )

            # Update best match if score is better
            if score > best_score:
                best_score = score
                best_person_id = person_id

        # Return match only if it meets the threshold
        if best_score >= self.similarity_threshold:
            return best_person_id

        return None

    def _create_new_person(
        self, face_embedding: np.ndarray, face_location: Tuple, frame_number: int
    ) -> str:
        """
        Create a new tracked person with a unique ID.

        """
        person_id = f"P{self.next_person_id:03d}"
        self.next_person_id += 1

        self.tracked_persons[person_id] = {
            "embedding": face_embedding.copy() if face_embedding is not None else None,
            "last_location": face_location,
            "last_seen_frame": frame_number,
            "first_seen_frame": frame_number,
            "appearances": 1,
        }

        return person_id

    def _update_person(
        self,
        person_id: str,
        face_embedding: np.ndarray,
        face_location: Tuple,
        frame_number: int,
        embedding_update_rate: float = 0.3,
    ):
        """
        Update tracked person information with new detection.
        """
        person_data = self.tracked_persons[person_id]

        # 1. ATUALIZAÇÃO DA MÉDIA (Para estabilidade no rastreamento frame-a-frame)
        if face_embedding is not None and person_data["embedding"] is not None:
            person_data["embedding"] = (1 - embedding_update_rate) * person_data[
                "embedding"
            ] + embedding_update_rate * face_embedding
        elif face_embedding is not None:
            person_data["embedding"] = face_embedding.copy()

        # 2. CAPTURA DO REPRESENTANTE (Para o Merge/Re-ID final) -- PARTE QUE FALTOU
        if face_location:
            top, right, bottom, left, _ = face_location

            # Calcula área do rosto (resolução)
            current_area = (bottom - top) * (right - left)

            # Se este rosto for maior que o melhor já visto neste track, ele vira o representante
            if current_area > person_data.get("best_face_area", 0):
                person_data["representative_embedding"] = face_embedding.copy()
                person_data["best_face_area"] = current_area

        # Update location and frame info
        person_data["last_location"] = face_location
        person_data["last_seen_frame"] = frame_number
        person_data["appearances"] += 1

    def _get_active_tracks(self, frame_number: int) -> List[Tuple[str, Dict]]:
        """
        Get list of active tracks that can be matched.

        Args:
            frame_number: Current frame number

        Returns:
            List of tuples (person_id, person_data) for active tracks
        """
        active_tracks = []
        for person_id, person_data in self.tracked_persons.items():
            frames_missing = frame_number - person_data["last_seen_frame"]
            if frames_missing <= self.max_frames_missing:
                active_tracks.append((person_id, person_data))
        return active_tracks

    def track_frame(
        self, persons: List[PersonInfo], frame_number: int
    ) -> List[PersonInfo]:
        """
        Track persons in a single frame and assign consistent IDs using Hungarian Algorithm.

        """
        # Separate valid detections from invalid ones
        valid_detections = []
        valid_indices = []

        for idx, person in enumerate(persons):
            if person["face_embedding"] is None or person["face_location"] is None:
                person["person_id"] = "Unknown"
            else:
                valid_detections.append(person)
                valid_indices.append(idx)

        if not valid_detections:
            return persons

        # Get active tracks
        active_tracks = self._get_active_tracks(frame_number)

        if not active_tracks:
            # No active tracks, create new persons for all detections
            for person in valid_detections:
                face_embedding = np.array(person["face_embedding"])
                face_location = person["face_location"]
                person_id = self._create_new_person(
                    face_embedding, face_location, frame_number
                )
                person["person_id"] = person_id
            return persons

        # Build cost matrix: rows = tracks, columns = detections
        # We use negative scores as costs (to minimize)
        num_tracks = len(active_tracks)
        num_detections = len(valid_detections)
        cost_matrix = np.zeros((num_tracks, num_detections))

        for i, (person_id, person_data) in enumerate(active_tracks):
            for j, detection in enumerate(valid_detections):
                face_embedding = np.array(detection["face_embedding"])
                face_location = detection["face_location"]

                # Calculate match score
                score = self._calculate_match_score(
                    face_embedding,
                    face_location,
                    person_data["embedding"],
                    person_data["last_location"],
                )

                # Use negative score as cost (minimize cost = maximize score)
                cost_matrix[i, j] = -score

        # Apply Hungarian Algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Track which detections have been matched
        matched_detections = set()

        # Process matches and filter by threshold
        for track_idx, det_idx in zip(track_indices, detection_indices):
            score = -cost_matrix[track_idx, det_idx]  # Convert back to positive score

            if score >= self.similarity_threshold:
                # Valid match
                person_id, person_data = active_tracks[track_idx]
                detection = valid_detections[det_idx]

                face_embedding = np.array(detection["face_embedding"])
                face_location = detection["face_location"]

                # Update existing person
                self._update_person(
                    person_id, face_embedding, face_location, frame_number
                )

                detection["person_id"] = person_id
                matched_detections.add(det_idx)

        # Create new persons for unmatched detections
        for j, detection in enumerate(valid_detections):
            if j not in matched_detections:
                face_embedding = np.array(detection["face_embedding"])
                face_location = detection["face_location"]
                person_id = self._create_new_person(
                    face_embedding, face_location, frame_number
                )
                detection["person_id"] = person_id

        return persons

    def cleanup_old_tracks(self, current_frame: int):
        to_remove = []
        for person_id, person_data in self.tracked_persons.items():
            frames_missing = current_frame - person_data["last_seen_frame"]
            # Se sumiu por muito tempo, movemos para o arquivo morto
            if frames_missing > self.max_frames_missing * 2:
                to_remove.append(person_id)

        for person_id in to_remove:
            # Move para o arquivo antes de deletar dos ativos
            self.archived_tracks[person_id] = self.tracked_persons[person_id]
            del self.tracked_persons[person_id]

    def finalize(self):
        """Chame isso no fim do vídeo para mover os restantes para o arquivo."""
        self.archived_tracks.update(self.tracked_persons)
        self.tracked_persons = {}

    def get_tracking_stats(self) -> Dict:
        """
        Get statistics about tracked persons.

        Returns:
            Dictionary with tracking statistics
        """
        return {
            "total_persons_tracked": len(self.tracked_persons),
            "next_person_id": self.next_person_id - 1,
            "active_persons": sum(
                1 for p in self.tracked_persons.values() if p["appearances"] > 1
            ),
        }


def merge_tracklets(
    all_tracks: Dict[str, Dict],
    similarity_threshold: float = 0.75,  # Um pouco mais rigoroso que o tracker frame-a-frame
) -> Dict[str, str]:
    """
    Analisa todos os tracks gerados e cria um mapa de 'De -> Para' para unificar IDs.
    Ex: {'P005': 'P001', 'P009': 'P001'}
    """

    # 1. Preparar dados: Ordenar tracks pelo tempo de início
    # Formato da lista: (id, start_frame, end_frame, representative_embedding)
    sorted_tracks = []
    for pid, data in all_tracks.items():
        # Assume que você salvou 'first_seen_frame' e 'last_seen_frame' no tracker
        # E que tem um 'representative_embedding' (ou a média dos embeddings)
        if "representative_embedding" not in data:
            continue  # Ignora tracks muito curtas sem embedding confiável

        sorted_tracks.append(
            {
                "id": pid,
                "start": data["first_seen_frame"],
                "end": data["last_seen_frame"],
                "embedding": data["representative_embedding"],
            }
        )

    # Ordena por quem apareceu primeiro
    sorted_tracks.sort(key=lambda x: x["start"])

    # Mapa de redirecionamento de IDs (ex: P05 -> P01)
    id_map = {t["id"]: t["id"] for t in sorted_tracks}

    # 2. Comparação Todos-contra-Todos (Otimizada)
    for i in range(len(sorted_tracks)):
        track_a = sorted_tracks[i]

        # Se este track já foi fundido a outro anterior, usamos o ID mestre dele
        root_id_a = id_map[track_a["id"]]

        for j in range(i + 1, len(sorted_tracks)):
            track_b = sorted_tracks[j]
            root_id_b = id_map[track_b["id"]]

            # Se já são a mesma pessoa, pula
            if root_id_a == root_id_b:
                continue

            # A. VERIFICAÇÃO TEMPORAL (CRUCIAL)
            # Se os tempos se sobrepõem, IMPOSSÍVEL ser a mesma pessoa
            # (Assumindo que o detector não detectou a mesma pessoa 2x no mesmo frame)
            overlap = max(
                0,
                min(track_a["end"], track_b["end"])
                - max(track_a["start"], track_b["start"]),
            )
            if overlap > 0:
                continue

            # B. VERIFICAÇÃO DE SIMILARIDADE
            sim = cosine_similarity(track_a["embedding"], track_b["embedding"])

            if sim > similarity_threshold:
                #  Se B já foi absorvido por alguém, pegue o mestre dele
                current_master_b = id_map[track_b["id"]]

                # Redireciona tudo que apontava para o B (ou mestre do B) para o A
                for key, val in id_map.items():
                    if val == current_master_b:
                        id_map[key] = root_id_a

    return id_map


def track_faces_in_video(frames: List[FrameInfo], **tracker_params) -> List[FrameInfo]:
    """
    Track faces across all frames in a video.

    """
    tracker = FaceTracker(**tracker_params)

    # Pass 1: Rastreamento linear (Gera P01, P02... P50)
    for i, frame in enumerate(frames):
        if "persons" in frame:
            frame["persons"] = tracker.track_frame(frame["persons"], i)
            tracker.cleanup_old_tracks(i)

    tracker.finalize()  # Move o que sobrou para o arquivo

    # Pass 2: Calcular fusões
    merge_map = merge_tracklets(tracker.archived_tracks)

    # Pass 3: Reescrever a história
    for frame in frames:
        if "persons" not in frame:
            continue

        for person in frame["persons"]:
            old_id = person.get("person_id")
            if old_id and old_id in merge_map:
                new_id = merge_map[old_id]
                person["person_id"] = new_id

    return frames
