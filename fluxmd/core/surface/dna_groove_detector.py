# fluxmd/core/surface/dna_groove_detector.py
"""DNA groove detection for biased surface hopping."""

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["DNAGrooveDetector"]


class DNAGrooveDetector:
    """Detect and label DNA major/minor grooves on molecular surfaces."""

    def __init__(self, atoms: Dict[str, np.ndarray]):
        """Initialize with molecular structure.

        Args:
            atoms: Dict with 'coords', 'names', 'resnames' arrays
        """
        self.atoms = atoms
        self._identify_dna_backbone()
        self._build_base_pairs()

    def _identify_dna_backbone(self) -> None:
        """Identify DNA backbone atoms and build spatial index."""
        # DNA residue names
        dna_residues = {"DA", "DT", "DG", "DC", "A", "T", "G", "C", "ADE", "THY", "GUA", "CYT"}

        # Backbone atom names
        backbone_atoms = {"P", "O5'", "C5'", "C4'", "C3'", "O3'", "OP1", "OP2"}

        # Find DNA backbone atoms
        backbone_mask = []
        self.backbone_indices = []

        for i, (name, resname) in enumerate(
            zip(self.atoms["names"], self.atoms.get("resnames", [""] * len(self.atoms["names"])))
        ):
            if resname.upper() in dna_residues and name.upper() in backbone_atoms:
                backbone_mask.append(True)
                self.backbone_indices.append(i)
            else:
                backbone_mask.append(False)

        self.backbone_indices = np.array(self.backbone_indices)
        self.n_backbone = len(self.backbone_indices)

        if self.n_backbone > 0:
            self.backbone_coords = self.atoms["coords"][self.backbone_indices]
            self.backbone_tree = cKDTree(self.backbone_coords)
            self.has_dna = True
        else:
            self.has_dna = False

        print(f"  Found {self.n_backbone} DNA backbone atoms")

    def _build_base_pairs(self) -> None:
        """Identify base pairs and helical axis."""
        if not self.has_dna:
            return

        # Simplified: Find C1' atoms to determine base positions
        c1_prime_indices = []
        for i in self.backbone_indices:
            if self.atoms["names"][i] == "C1'":
                c1_prime_indices.append(i)

        if len(c1_prime_indices) < 2:
            self.base_pairs = []
            return

        # Simple pairing: assume antiparallel strands
        # In reality, would use more sophisticated base pairing detection
        c1_coords = self.atoms["coords"][c1_prime_indices]

        # Find pairs by distance (~10.5 Å for Watson-Crick pairs)
        pairs = []
        used = set()

        for i, coord1 in enumerate(c1_coords):
            if i in used:
                continue

            distances = np.linalg.norm(c1_coords - coord1, axis=1)
            # Watson-Crick distance range
            candidates = np.where((distances > 9.0) & (distances < 12.0))[0]

            if len(candidates) > 0:
                # Take closest in range
                j = candidates[np.argmin(distances[candidates])]
                if j not in used:
                    pairs.append((c1_prime_indices[i], c1_prime_indices[j]))
                    used.add(i)
                    used.add(j)

        self.base_pairs = pairs
        print(f"  Found {len(self.base_pairs)} base pairs")

    def _calculate_groove_direction(
        self, point: np.ndarray, backbone_indices: List[int]
    ) -> Tuple[np.ndarray, str]:
        """Calculate groove direction at a point near DNA.

        Args:
            point: 3D position to evaluate
            backbone_indices: Nearby backbone atom indices

        Returns:
            (groove_direction, groove_type) where type is 'major' or 'minor'
        """
        if len(backbone_indices) < 3:
            return np.array([0, 0, 1]), "none"

        # Get phosphate positions (simplified groove detection)
        phosphates = []
        for idx in backbone_indices:
            atom_idx = self.backbone_indices[idx]
            if self.atoms["names"][atom_idx] == "P":
                phosphates.append(self.atoms["coords"][atom_idx])

        if len(phosphates) < 2:
            return np.array([0, 0, 1]), "none"

        phosphates = np.array(phosphates)

        # Estimate local helical axis (simplified)
        if len(phosphates) >= 3:
            # Use three consecutive phosphates
            p1, p2, p3 = phosphates[:3]

            # Vector along backbone
            v1 = p2 - p1
            v2 = p3 - p2

            # Approximate helical axis as average direction
            axis = (v1 + v2) / 2
            axis /= np.linalg.norm(axis)

            # Vector from DNA center to point
            dna_center = np.mean(phosphates, axis=0)
            radial = point - dna_center
            radial -= np.dot(radial, axis) * axis  # Project to plane perpendicular to axis

            if np.linalg.norm(radial) < 1e-6:
                return axis, "none"

            radial /= np.linalg.norm(radial)

            # Calculate angle in helical plane
            # This is simplified - real implementation would use proper helical parameters
            # For B-DNA: major groove ~-120° to -60°, minor groove ~60° to 120°

            # Create reference frame
            ref_point = phosphates[0]
            ref_vec = ref_point - dna_center
            ref_vec -= np.dot(ref_vec, axis) * axis
            ref_vec /= np.linalg.norm(ref_vec)

            # Cross product for perpendicular
            perp = np.cross(axis, ref_vec)

            # Calculate angle
            cos_angle = np.dot(radial, ref_vec)
            sin_angle = np.dot(radial, perp)
            angle = np.arctan2(sin_angle, cos_angle)
            angle_deg = np.degrees(angle)

            # Classify groove (simplified for B-DNA)
            # These ranges are approximate and would need refinement
            if -150 <= angle_deg <= -30:
                groove_type = "major"
            elif 30 <= angle_deg <= 150:
                groove_type = "minor"
            else:
                groove_type = "none"

            return radial, groove_type

        return np.array([0, 0, 1]), "none"

    def label_surface_vertices(
        self, vertices: np.ndarray, radius_cutoff: float = 5.0
    ) -> Dict[int, str]:
        """Label surface vertices by groove type.

        Args:
            vertices: (N, 3) array of surface vertex positions
            radius_cutoff: Maximum distance to DNA backbone

        Returns:
            Dict mapping vertex index to groove type ('major', 'minor', 'none')
        """
        if not self.has_dna:
            return {}

        vertex_labels = {}

        for i, vertex in enumerate(vertices):
            # Find nearby backbone atoms
            distances, indices = self.backbone_tree.query(
                vertex, k=min(10, self.n_backbone), distance_upper_bound=radius_cutoff
            )

            # Filter valid indices
            valid_mask = distances < radius_cutoff
            valid_indices = indices[valid_mask]

            if len(valid_indices) > 0:
                _, groove_type = self._calculate_groove_direction(vertex, valid_indices)
                vertex_labels[i] = groove_type
            else:
                vertex_labels[i] = "none"

        # Count groove vertices
        major_count = sum(1 for v in vertex_labels.values() if v == "major")
        minor_count = sum(1 for v in vertex_labels.values() if v == "minor")

        print(
            f"  Groove labeling: {major_count} major, {minor_count} minor, "
            f"{len(vertices) - major_count - minor_count} other"
        )

        return vertex_labels

    def get_groove_bias(
        self, position: np.ndarray, target_position: np.ndarray, groove_preference: str = "major"
    ) -> float:
        """Calculate groove bias factor for trajectory hopping.

        Args:
            position: Current position
            target_position: Proposed hop position
            groove_preference: Preferred groove type ('major' or 'minor')

        Returns:
            Bias factor [0, 1] where 1 is maximum preference
        """
        if not self.has_dna:
            return 1.0

        # Find nearby DNA
        dist, idx = self.backbone_tree.query(target_position)

        if dist > 8.0:  # Too far from DNA
            return 1.0

        # Get nearby backbone atoms
        distances, indices = self.backbone_tree.query(
            target_position, k=min(10, self.n_backbone), distance_upper_bound=8.0
        )

        valid_mask = distances < 8.0
        valid_indices = indices[valid_mask]

        if len(valid_indices) == 0:
            return 1.0

        # Calculate groove direction
        groove_dir, groove_type = self._calculate_groove_direction(target_position, valid_indices)

        # Base bias on groove type match
        if groove_type == groove_preference:
            type_bias = 1.0
        elif groove_type == "none":
            type_bias = 0.7
        else:
            type_bias = 0.4  # Wrong groove

        # Additional bias based on movement direction alignment
        hop_direction = target_position - position
        hop_direction /= np.linalg.norm(hop_direction)

        # Prefer hops along groove direction
        direction_alignment = abs(np.dot(hop_direction, groove_dir))
        direction_bias = 0.5 + 0.5 * direction_alignment

        return type_bias * direction_bias
