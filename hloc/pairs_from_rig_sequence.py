"""Generate COLMAP-style sequential pairs for multi-camera rigs.

The rig configuration uses COLMAP's JSON format.  Images below each camera's
``image_prefix`` are ordered by their remaining path, and equal remaining paths
denote synchronized measurements in the same frame.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from . import logger
from .pairs_from_retrieval import get_descriptors, pairs_from_score_matrix


Pair = Tuple[str, str]


def _canonical_pair(name0: str, name1: str) -> Pair:
    if name0 == name1:
        raise ValueError("An image cannot be paired with itself.")
    return tuple(sorted((name0, name1)))  # type: ignore[return-value]


def _read_config(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, list) or not config:
        raise ValueError("A rig config must be a non-empty list of rigs.")
    return config


def _frame_name_for_camera(name: str, prefix: str) -> Optional[str]:
    """Return an image's frame name when it belongs to a rig camera.

    Image processing may move a camera stream into a grouping folder such as
    ``NewGroup_Cam01``.  A COLMAP rig config still identifies that stream as
    ``Cam01/``.  In addition to the configured prefix, accept a path component
    named ``CAMERA`` or ``*_CAMERA`` and use the path after that component as
    the frame name.
    """
    if name.startswith(prefix):
        return name[len(prefix) :]

    camera_name = Path(prefix.rstrip("/")).name
    # if "09" in camera_name:
    #     print(camera_name)
    if not camera_name:
        return None
    parts = Path(name).parts
    # if "09" in camera_name:
    #     print(parts)
    matches = [
        index
        for index, part in enumerate(parts)
        if part == camera_name or part.endswith(f"_{camera_name}")
    ]
    if len(matches) != 1:
        return None
    frame_parts = parts[matches[0] + 1 :]
    # print(Path(*frame_parts).as_posix() if frame_parts else "")
    return Path(*frame_parts).as_posix() if frame_parts else ""


def rig_sequence_pairs(
    image_list: Sequence[str],
    rig_config: Union[Path, str],
    overlap: int = 10,
    quadratic_overlap: bool = True,
) -> Set[Pair]:
    """Return temporal and synchronized cross-camera pairs for configured rigs."""
    if overlap < 1:
        raise ValueError("Rig sequential overlap must be at least one.")

    names = sorted(set(image_list))
    logger.info(f"image cameras: {len(names)}")
    if len(names) != len(image_list):
        raise ValueError("Rig image names must be unique.")
    config = _read_config(Path(rig_config))
    assigned: Dict[str, Tuple[int, int, str]] = {}
    streams: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    frames: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    frame_cameras: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    reference_cameras: Dict[int, int] = {}
    configured_cameras: Dict[int, int] = {}

    for rig_index, rig in enumerate(config):
        cameras = rig.get("cameras") if isinstance(rig, dict) else None
        logger.info(f"Camera names: {cameras}")
        if not isinstance(cameras, list) or len(cameras) < 2:
            raise ValueError(f"Rig {rig_index} must configure at least two cameras.")
        if sum(bool(camera.get("ref_sensor", False)) for camera in cameras) != 1:
            raise ValueError(f"Rig {rig_index} must configure exactly one reference sensor.")
        configured_cameras[rig_index] = len(cameras)
        for camera_index, camera in enumerate(cameras):
            if camera.get("ref_sensor", False):
                reference_cameras[rig_index] = camera_index
            prefix = camera.get("image_prefix") if isinstance(camera, dict) else None
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(f"Rig {rig_index} camera {camera_index} needs an image_prefix.")
            
            for name in names:
                frame_name = _frame_name_for_camera(name, prefix)
                # logger.info(f"Frame name for {name}: {frame_name} in prefix {prefix}")
                if frame_name is not None:
                    if name in assigned:
                        raise ValueError(f"Image {name} matches multiple rig camera prefixes.")
                    if not frame_name:
                        raise ValueError(f"Image {name} has no frame name after prefix {prefix}.")
                    assigned[name] = (rig_index, camera_index, frame_name)
                    streams[(rig_index, camera_index)].append((frame_name, name))
                    frames[(rig_index, frame_name)].append(name)
                    frame_cameras[(rig_index, frame_name)].add(camera_index)

    missing = sorted(set(names) - set(assigned))
    if missing:
        raise ValueError(f"Rig config does not assign images: {', '.join(missing[:5])}")

    for rig_index, camera_count in configured_cameras.items():
        reference = reference_cameras[rig_index]
        for camera_index in range(camera_count):
            if not streams[(rig_index, camera_index)]:
                raise ValueError(f"Rig {rig_index} camera {camera_index} has no matching images.")
            if camera_index != reference and not any(
                reference in cameras and camera_index in cameras
                for (frame_rig, _), cameras in frame_cameras.items()
                if frame_rig == rig_index
            ):
                raise ValueError(
                    f"Rig {rig_index} camera {camera_index} has no synchronized frame with the reference sensor."
                )

    pairs: Set[Pair] = set()
    for frame_images in frames.values():
        pairs.update(_canonical_pair(a, b) for a, b in combinations(sorted(frame_images), 2))

    for stream in streams.values():
        ordered = [name for _, name in sorted(stream)]
        for index, name in enumerate(ordered):
            offsets = range(1, overlap + 1)
            for offset in offsets:
                if index + offset < len(ordered):
                    pairs.add(_canonical_pair(name, ordered[index + offset]))
                if quadratic_overlap and index + offset * offset < len(ordered):
                    pairs.add(_canonical_pair(name, ordered[index + offset * offset]))
    return pairs


def retrieval_pairs(image_list: Sequence[str], descriptors: Path, num_matched: int) -> Set[Pair]:
    if num_matched < 1:
        raise ValueError("The number of retrieval loop closures must be at least one.")
    names = list(image_list)
    if len(names) < 2:
        return set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    descriptors_tensor = get_descriptors(names, descriptors).to(device)
    scores = torch.einsum("id,jd->ij", descriptors_tensor, descriptors_tensor)
    invalid = np.eye(len(names), dtype=bool)
    selected = pairs_from_score_matrix(scores, invalid, min(num_matched, len(names) - 1), min_score=0)
    pairs: Set[Pair] = set()
    for i, j in selected:
        pairs.add(_canonical_pair(names[i], names[j]))
    logger.info("Found %d retrieval loop closure pairs.", len(pairs))
    return pairs
    # return {_canonical_pair(names[i], names[j]) for i, j in selected}


def main(
    output: Path,
    image_list: Sequence[str],
    rig_config: Union[Path, str],
    overlap: int = 10,
    quadratic_overlap: bool = True,
    retrieval_descriptors: Optional[Path] = None,
    retrieval_loop_closures: bool = True,
    loop_closure_num_matched: int = 50,
) -> Set[Pair]:
    pairs = rig_sequence_pairs(image_list, rig_config, overlap, quadratic_overlap)
    logger.info("Found %d rig sequential pairs.", len(pairs))
    if retrieval_loop_closures:
        if retrieval_descriptors is None:
            raise ValueError("Retrieval descriptors are required when loop closures are enabled.")
        pairs = pairs | retrieval_pairs(image_list, retrieval_descriptors, loop_closure_num_matched)
    logger.info("A total of %d rig sequential pairs have been found.", len(pairs))
    ordered_pairs = sorted(pairs)
    with open(output, "w", encoding="utf-8") as file:
        file.write("\n".join(" ".join(pair) for pair in ordered_pairs))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rig_config", required=True, type=Path)
    parser.add_argument("--image_list", required=True, type=Path)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--no_quadratic_overlap", action="store_true")
    parser.add_argument("--retrieval_descriptors", type=Path)
    parser.add_argument("--no_retrieval_loop_closures", action="store_true")
    parser.add_argument("--loop_closure_num_matched", type=int, default=50)
    args = parser.parse_args()
    names = [line.strip() for line in args.image_list.read_text().splitlines() if line.strip()]
    main(
        args.output,
        names,
        args.rig_config,
        args.overlap,
        not args.no_quadratic_overlap,
        args.retrieval_descriptors,
        not args.no_retrieval_loop_closures,
        args.loop_closure_num_matched,
    )
