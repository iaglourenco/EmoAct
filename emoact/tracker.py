"""
Face tracking module for maintaining consistent person IDs across video frames.
Uses face embeddings and spatial proximity to track individuals.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
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

        # Update embedding with exponential moving average
        if face_embedding is not None and person_data["embedding"] is not None:
            person_data["embedding"] = (1 - embedding_update_rate) * person_data[
                "embedding"
            ] + embedding_update_rate * face_embedding
        elif face_embedding is not None:
            person_data["embedding"] = face_embedding.copy()

        # Update location and frame info
        person_data["last_location"] = face_location
        person_data["last_seen_frame"] = frame_number
        person_data["appearances"] += 1

    def track_frame(
        self, persons: List[PersonInfo], frame_number: int
    ) -> List[PersonInfo]:
        """
        Track persons in a single frame and assign consistent IDs.

        """
        for person in persons:
            if person["face_embedding"] is None or person["face_location"] is None:
                person["person_id"] = "Unknown"
                continue

            face_embedding = np.array(person["face_embedding"])
            face_location = person["face_location"]

            # Try to match with existing tracked person
            person_id = self._find_best_match(
                face_embedding, face_location, frame_number
            )

            if person_id is not None:
                # Update existing person
                self._update_person(
                    person_id, face_embedding, face_location, frame_number
                )
            else:
                # Create new person
                person_id = self._create_new_person(
                    face_embedding, face_location, frame_number
                )

            person["person_id"] = person_id

        return persons

    def cleanup_old_tracks(self, current_frame: int):
        """
        Remove tracked persons that haven't been seen recently.

        """
        to_remove = []
        for person_id, person_data in self.tracked_persons.items():
            frames_missing = current_frame - person_data["last_seen_frame"]
            if frames_missing > self.max_frames_missing * 2:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.tracked_persons[person_id]

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


def track_faces_in_video(frames: List[FrameInfo], **tracker_params) -> List[FrameInfo]:
    """
    Track faces across all frames in a video.

    """
    tracker = FaceTracker(**tracker_params)

    for frame_idx, frame_info in enumerate(frames):
        if "persons" in frame_info and len(frame_info["persons"]) > 0:
            frame_info["persons"] = tracker.track_frame(
                frame_info["persons"], frame_idx
            )

    return frames
