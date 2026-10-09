from datetime import datetime
from pathlib import Path

import cv2
import json
import numpy as np
import supervision as sv
import torch

from tqdm import tqdm
from ultralytics import YOLO


START_FRAME = 500
MAX_FRAME = 600

VIDEO = "videos/test-qair.mp4"
MODEL = "yolo26m.pt"

DEVICE = 0  # 0 = GPU, "cpu" = CPU

BATCH_SIZE = 1

TILE_SIZE = 320
OVERLAP = 0.20

CONF = 0.05
NMS_IOU = 0.50

BIRD_CLASS = 14


# ---------------------------------------------------------
# ByteTrack
# ---------------------------------------------------------

TRACK_ACTIVATION_THRESHOLD = 0.05
LOST_TRACK_BUFFER = 45
MINIMUM_MATCHING_THRESHOLD = 0.60
MINIMUM_CONSECUTIVE_FRAMES = 1


def print_device_info():
    if DEVICE == "cpu":
        print("Device : CPU")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU demandé mais CUDA n'est pas disponible."
        )

    print("CUDA disponible : oui")
    print(f"GPU : {torch.cuda.get_device_name(DEVICE)}")

    print(
        f"VRAM : "
        f"{torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3:.1f} Go"
    )


def create_output_dir():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    path = Path("run") / MODEL.replace(".pt", "") / now

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path, now


def create_video_writer(
    cap,
    fps,
    output_path,
):
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    video_path = (
        output_path / "tracked.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Impossible de créer la vidéo : "
            f"{video_path}"
        )

    return writer, video_path


def open_video():
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la vidéo : {VIDEO}"
        )

    return cap


def get_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    return fps, total_frames


def print_config(now, fps, output_path):
    print(
        f"Début de la détection d'oiseaux : {now}"
    )

    print(f"Vidéo : {VIDEO}")
    print(f"Modèle : {MODEL}")

    print(
        f"Entre les frames {START_FRAME} "
        f"et {MAX_FRAME}."
    )

    print(
        f"Taille des tuiles : {TILE_SIZE} pixels, "
        f"chevauchement : {OVERLAP * 100:.0f}%"
    )

    print(f"Confiance minimum : {CONF}")
    print(f"NMS IoU : {NMS_IOU}")

    print(
        f"ByteTrack activation : "
        f"{TRACK_ACTIVATION_THRESHOLD}"
    )

    print(
        f"ByteTrack lost buffer : "
        f"{LOST_TRACK_BUFFER}"
    )

    print(
        f"ByteTrack matching threshold : "
        f"{MINIMUM_MATCHING_THRESHOLD}"
    )

    print(
        f"Fréquence d'images : "
        f"{fps:.3f} fps"
    )

    print(
        f"Résultats : {output_path}"
    )


def detect_on_tiles(
    model,
    frame,
    frame_index,
    timestamp_seconds,
):
    height, width = frame.shape[:2]

    step = int(
        TILE_SIZE * (1 - OVERLAP)
    )

    tiles = []
    tile_positions = []

    # -----------------------------------------------------
    # Création des tiles
    # -----------------------------------------------------

    for y in range(0, height, step):
        for x in range(0, width, step):

            tile_x2 = min(
                x + TILE_SIZE,
                width,
            )

            tile_y2 = min(
                y + TILE_SIZE,
                height,
            )

            tile = frame[
                y:tile_y2,
                x:tile_x2
            ]

            # Ignore petites tiles de bord.
            if (
                tile.shape[0] < 200
                or tile.shape[1] < 200
            ):
                continue

            tiles.append(tile)

            tile_positions.append(
                {
                    "x": x,
                    "y": y,
                }
            )

    detections = []

    # -----------------------------------------------------
    # Inférence par batch
    # -----------------------------------------------------

    for batch_start in range(
        0,
        len(tiles),
        BATCH_SIZE,
    ):

        batch_tiles = tiles[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_positions = tile_positions[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        results = model.predict(
            batch_tiles,
            imgsz=640,
            conf=CONF,
            classes=[BIRD_CLASS],
            device=DEVICE,
            verbose=False,
        )

        # -------------------------------------------------
        # Résultats
        # -------------------------------------------------

        for result, position in zip(
            results,
            batch_positions,
        ):

            tile_x = position["x"]
            tile_y = position["y"]

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                detections.append(
                    {
                        "frame": frame_index,

                        "timestamp":
                            timestamp_seconds,

                        "class_id":
                            class_id,

                        "class_name":
                            model.names[
                                class_id
                            ],

                        # Tile -> frame globale
                        "x1": x1 + tile_x,
                        "y1": y1 + tile_y,
                        "x2": x2 + tile_x,
                        "y2": y2 + tile_y,

                        "confidence":
                            confidence,
                    }
                )

    return detections


def apply_global_nms(detections):
    if not detections:
        return []

    boxes = []
    scores = []

    for detection in detections:

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]

        width = x2 - x1
        height = y2 - y1

        boxes.append(
            [
                float(x1),
                float(y1),
                float(width),
                float(height),
            ]
        )

        scores.append(
            float(
                detection["confidence"]
            )
        )

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=CONF,
        nms_threshold=NMS_IOU,
    )

    if len(indices) == 0:
        return []

    filtered = []

    for index in indices:

        if hasattr(index, "__len__"):
            index = index[0]

        filtered.append(
            detections[int(index)]
        )

    return filtered


# ---------------------------------------------------------
# Conversion detections -> Supervision
# ---------------------------------------------------------

def detections_to_supervision(detections):
    if not detections:
        return sv.Detections.empty()

    xyxy = np.array(
        [
            [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ]
            for detection in detections
        ],
        dtype=np.float32,
    )

    confidence = np.array(
        [
            detection["confidence"]
            for detection in detections
        ],
        dtype=np.float32,
    )

    class_id = np.array(
        [
            detection["class_id"]
            for detection in detections
        ],
        dtype=int,
    )

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


# ---------------------------------------------------------
# ByteTrack
# ---------------------------------------------------------

def track_detections(
    tracker,
    detections,
    frame_index,
    timestamp_seconds,
):
    sv_detections = (
        detections_to_supervision(
            detections
        )
    )

    tracked = (
        tracker.update_with_detections(
            sv_detections
        )
    )

    tracked_detections = []

    if len(tracked) == 0:
        return tracked_detections

    for i in range(len(tracked)):

        x1, y1, x2, y2 = (
            tracked.xyxy[i]
        )

        tracker_id = (
            tracked.tracker_id[i]
        )

        confidence = (
            tracked.confidence[i]
            if tracked.confidence is not None
            else 0.0
        )

        class_id = (
            tracked.class_id[i]
            if tracked.class_id is not None
            else BIRD_CLASS
        )

        tracked_detections.append(
            {
                "frame":
                    frame_index,

                "timestamp":
                    timestamp_seconds,

                "track_id":
                    int(tracker_id),

                "class_id":
                    int(class_id),

                "class_name":
                    "bird",

                "x1":
                    float(x1),

                "y1":
                    float(y1),

                "x2":
                    float(x2),

                "y2":
                    float(y2),

                "confidence":
                    float(confidence),
            }
        )

    return tracked_detections


def draw_detections(
    frame,
    detections,
):
    for detection in detections:

        x1 = int(
            detection["x1"]
        )

        y1 = int(
            detection["y1"]
        )

        x2 = int(
            detection["x2"]
        )

        y2 = int(
            detection["y2"]
        )

        track_id = detection.get(
            "track_id"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        if track_id is not None:

            label = (
                f"bird #{track_id} "
                f"{detection['confidence']:.2f}"
            )

        else:

            label = (
                f"bird "
                f"{detection['confidence']:.2f}"
            )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(20, y1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def save_frame(
    frame,
    frame_index,
    output_path,
):
    filename = (
        output_path
        / f"frame_{frame_index}.jpg"
    )

    cv2.imwrite(
        str(filename),
        frame,
    )


def save_detections(
    detections,
    output_path,
):
    filename = (
        output_path
        / "detections.json"
    )

    with open(
        filename,
        "w",
    ) as file:

        json.dump(
            detections,
            file,
            indent=2,
        )


def main():

    print_device_info()

    output_path, now = create_output_dir()

    model = YOLO(MODEL)

    cap = open_video()

    fps, total_video_frames = get_video_info(cap)

    # -----------------------------------------------------
    # Création de la vidéo de sortie
    # -----------------------------------------------------

    video_writer, video_output_path = create_video_writer(
        cap=cap,
        fps=fps,
        output_path=output_path,
    )

    # -----------------------------------------------------
    # Création du tracker
    # -----------------------------------------------------

    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
        frame_rate=round(fps),
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
    )

    print_config(
        now=now,
        fps=fps,
        output_path=output_path,
    )

    end_frame = min(
        MAX_FRAME,
        total_video_frames - 1,
    )

    if START_FRAME > end_frame:
        raise ValueError(
            f"START_FRAME ({START_FRAME}) "
            f"dépasse la fin de la vidéo "
            f"({total_video_frames} frames)."
        )

    # Positionnement sur la première frame à analyser.
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        START_FRAME,
    )

    frame_index = START_FRAME

    frames_to_process = (
        end_frame
        - START_FRAME
        + 1
    )

    all_detections = []
    unique_track_ids = set()

    with tqdm(
        total=frames_to_process,
        desc=f"YOLO {MODEL}",
        unit="frame",
    ) as progress:

        while frame_index <= end_frame:

            ok, frame = cap.read()

            if not ok:
                break

            timestamp_seconds = (
                frame_index / fps
            )

            # ---------------------------------------------
            # 1. YOLO sur les tiles
            # ---------------------------------------------

            raw_detections = detect_on_tiles(
                model=model,
                frame=frame,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )

            # ---------------------------------------------
            # 2. NMS global
            # ---------------------------------------------

            detections = apply_global_nms(
                raw_detections
            )

            # ---------------------------------------------
            # 3. ByteTrack
            #
            # Appelé à CHAQUE frame, même sans détection.
            # ---------------------------------------------

            tracked_detections = track_detections(
                tracker=tracker,
                detections=detections,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )

            # ---------------------------------------------
            # 4. Tracks présents sur cette frame
            # ---------------------------------------------

            if tracked_detections:

                all_detections.extend(
                    tracked_detections
                )

                for detection in tracked_detections:
                    unique_track_ids.add(
                        detection["track_id"]
                    )

                tqdm.write(
                    f"frame {frame_index}: "
                    f"{len(raw_detections)} raw -> "
                    f"{len(detections)} NMS -> "
                    f"{len(tracked_detections)} trackée(s) "
                    f"à {timestamp_seconds:.2f} s "
                    f"| IDs: "
                    f"{sorted([
                        d['track_id']
                        for d in tracked_detections
                    ])}"
                )

                # Dessine uniquement les tracks validés.
                draw_detections(
                    frame=frame,
                    detections=tracked_detections,
                )

                # JPG seulement pour les frames
                # contenant au moins un track.
                save_frame(
                    frame=frame,
                    frame_index=frame_index,
                    output_path=output_path,
                )

            # ---------------------------------------------
            # 5. Vidéo annotée
            #
            # TRÈS IMPORTANT :
            # cette ligne reste HORS du if.
            #
            # Ainsi toutes les frames sont écrites,
            # y compris celles sans oiseau.
            # ---------------------------------------------

            video_writer.write(frame)

            frame_index += 1

            progress.update(1)

    # -----------------------------------------------------
    # Fermeture des fichiers
    # -----------------------------------------------------

    cap.release()

    video_writer.release()

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    save_detections(
        detections=all_detections,
        output_path=output_path,
    )

    # -----------------------------------------------------
    # Résumé
    # -----------------------------------------------------

    print(
        f"\nTerminé : "
        f"{len(all_detections)} "
        f"détection(s) trackée(s)."
    )

    print(
        f"Nombre d'IDs ByteTrack créés : "
        f"{len(unique_track_ids)}"
    )

    print(
        f"IDs : "
        f"{sorted(unique_track_ids)}"
    )

    print(
        f"Résultats JSON : "
        f"{output_path / 'detections.json'}"
    )

    print(
        f"Vidéo trackée : "
        f"{video_output_path}"
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()