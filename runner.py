# -*- coding: utf-8 -*-
"""
XGI Wavefront Sensor - Streaming Pipeline Runner

Three-stage architecture:
1. Data Reading   - Simulate data stream at specified frame rate
2. Data Processing - Pipeline processes each frame
3. Data Saving    - Asynchronously save processing results

Usage:
    python runner.py                    # Default 10Hz, run indefinitely
    python runner.py --fps 20           # 20Hz
    python runner.py --duration 60      # Run for 60 seconds
"""

import sys
from pathlib import Path
import threading
import queue
import signal
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import task
from params import get_params


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class FrameData:
    """Container for a single frame."""

    frame_id: int
    image: np.ndarray


@dataclass
class ProcessedData:
    """Container for processed data."""

    frame_id: int
    wavefront_data: Optional[Dict] = None
    aberration_data: Optional[Dict] = None
    focus_data: Optional[Dict] = None
    success: bool = True


# =============================================================================
# Part 1: Data Reading
# =============================================================================


class DataReader(threading.Thread):
    """Read/simulate data stream at specified frame rate."""

    def __init__(
        self,
        image: np.ndarray,
        frame_queue: queue.Queue,
        fps: float = 10.0,
        duration: float = None,
    ):
        super().__init__(daemon=True)
        self.image = image
        self.frame_queue = frame_queue
        self.fps = fps
        self.duration = duration
        self.frame_interval = 1.0 / fps
        self._stop_event = threading.Event()
        self.total_frames = 0
        self.dropped_frames = 0

    def run(self):
        start_time = time.perf_counter()
        frame_id = 0

        while not self._stop_event.is_set():
            if self.duration and (time.perf_counter() - start_time) >= self.duration:
                break

            target_time = start_time + frame_id * self.frame_interval
            now = time.perf_counter()
            if target_time > now:
                time.sleep(target_time - now)

            frame = FrameData(frame_id=frame_id, image=self.image.copy())

            # Increment frame_id regardless of queue success
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.dropped_frames += 1

            frame_id += 1

        self.total_frames = frame_id
        self.frame_queue.put(None)

    def stop(self):
        self._stop_event.set()


# =============================================================================
# Part 2: Data Processing
# =============================================================================


class DataProcessor:
    """Process single frame data."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def process(self, frame: FrameData) -> ProcessedData:
        wavefront_data = None
        aberration_data = None
        focus_data = None

        try:
            for checkpoint_name, data in task(
                self.params,
                verbose=False,
                show_plots=False,
                do_rotation=self.params.get("do_rotation", False),
                parallel=self.params.get("parallel", True),
                img=frame.image,
                crop_size=self.params.get("crop_size", 2048),
                lowpass_cutoff=self.params.get("lowpass_cutoff", 0.35),
            ):
                if checkpoint_name == "checkpoint_wavefront":
                    wavefront_data = data
                elif checkpoint_name == "checkpoint_aberration":
                    aberration_data = data
                elif checkpoint_name == "checkpoint_focus":
                    focus_data = data

            return ProcessedData(
                frame_id=frame.frame_id,
                wavefront_data=wavefront_data,
                aberration_data=aberration_data,
                focus_data=focus_data,
                success=True,
            )
        except Exception as e:
            print(f"[Frame {frame.frame_id:04d}] ✗ Error: {e}")
            return ProcessedData(frame_id=frame.frame_id, success=False)


# =============================================================================
# Part 3: Data Saving
# =============================================================================


class DataSaver(threading.Thread):
    """Asynchronously save processing results."""

    def __init__(self, output_dir: Path):
        super().__init__(daemon=True)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self.saved_count = 0

    def run(self):
        while not self._stop_event.is_set():
            try:
                data = self._queue.get(timeout=1.0)
                if data is None:
                    break
                self._save(data)
                self.saved_count += 1
            except queue.Empty:
                continue

    def save(self, data: ProcessedData):
        self._queue.put(data)

    def _save(self, data: ProcessedData):
        frame_dir = self.output_dir / f"frame_{data.frame_id:06d}"
        frame_dir.mkdir(exist_ok=True)

        params_dict = {}

        # =========================================================
        # Checkpoint 1: wavefront_data
        # =========================================================
        if data.wavefront_data:
            wd = data.wavefront_data

            # npy: phase, fitted_phase, phase_error
            # if wd.get("phase") is not None:
            #     np.save(frame_dir / "phase.npy", wd["phase"].astype(np.float32))
            # if wd.get("fitted_phase") is not None:
            #     np.save(
            #         frame_dir / "fitted_phase.npy",
            #         wd["fitted_phase"].astype(np.float32),
            #     )
            if wd.get("phase_error") is not None:
                np.save(
                    frame_dir / "phase_error.npy", wd["phase_error"].astype(np.float32)
                )

            # params: fit_params, virtual_pixel_size, wavelength
            if wd.get("fit_params") is not None:
                fp = wd["fit_params"]
                if len(fp) >= 5:
                    params_dict["fit_x0_m"] = fp[0]
                    params_dict["fit_y0_m"] = fp[1]
                    params_dict["fit_Rx_m"] = fp[2]
                    params_dict["fit_Ry_m"] = fp[3]
                    params_dict["fit_A_rad"] = fp[4]
            if wd.get("virtual_pixel_size") is not None:
                vps = wd["virtual_pixel_size"]
                params_dict["virtual_pixel_size_y_m"] = vps[0]
                params_dict["virtual_pixel_size_x_m"] = vps[1]
            if wd.get("wavelength") is not None:
                params_dict["wavelength_m"] = wd["wavelength"]

        # =========================================================
        # Checkpoint 2: aberration_data
        # =========================================================
        if data.aberration_data:
            ad = data.aberration_data

            # txt: zernike_results
            zernike = ad.get("zernike_results")
            if zernike:
                if zernike.get("residual") is not None:
                    np.save(
                        frame_dir / "residual.npy", zernike["residual"].astype(np.float32)
                    )
                with open(frame_dir / "zernike_results.txt", "w") as f:
                    f.write("# Zernike Analysis Results\n")
                    f.write(f"rms_error_rad: {zernike.get('rms_error', 0):.6f}\n")
                    # f.write(
                    #     f"fit_residual_rad: {zernike.get('fit_residual', 0):.6f}\n\n"
                    # )

                    aberr = zernike.get("aberration_analysis", {})
                    if aberr:
                        f.write("# Aberration Coefficients\n")
                        f.write("# Index\tName\tRMS(nm)\n")
                        for key in sorted(aberr.keys(), key=lambda x: int(x[1:])):
                            ab = aberr[key]
                            f.write(f"{key}\t{ab['name']}\t{ab['rms_nm']}\n")

            # params: calibration_result
            cal = ad.get("calibration_result")
            if cal:
                params_dict["cal_R_x_m"] = cal.get("R_x", 0)
                params_dict["cal_R_y_m"] = cal.get("R_y", 0)
                params_dict["cal_R_avg_m"] = cal.get("R_avg", cal.get("R", 0))
                params_dict["cal_delta_R_x_m"] = cal.get("Delta_x", 0)
                params_dict["cal_delta_R_y_m"] = cal.get("Delta_y", 0)
                params_dict["source_distance"] = cal.get("source_distance", 0)

            # params: roi_result
            roi = ad.get("roi_result")
            if roi:
                params_dict["roi_center_x_m"] = roi.get("aperture_center", (0, 0))[0]
                params_dict["roi_center_y_m"] = roi.get("aperture_center", (0, 0))[1]
                params_dict["roi_radius"] = roi.get("aperture_radius_fraction", 0)

        # =========================================================
        # Checkpoint 3: focus_data
        # =========================================================
        if data.focus_data:
            fd = data.focus_data

            # npy: focus_field
            if fd.get("focus_field") is not None:
                np.save(
                    frame_dir / "focus_field.npy",
                    fd["focus_field"].astype(np.complex64),
                )

            # params: beam_position, beam_size, focus_position, focus_size, dx_focus, dy_focus
            bp = fd.get("beam_position", (0, 0))
            params_dict["beam_position_x_m"] = bp[0]
            params_dict["beam_position_y_m"] = bp[1]

            bs = fd.get("beam_size", {})
            params_dict["beam_fwhm_x_m"] = bs.get("fwhm_x", 0)
            params_dict["beam_fwhm_y_m"] = bs.get("fwhm_y", 0)

            fp = fd.get("focus_position", (0, 0))
            params_dict["focus_position_x_m"] = fp[0]
            params_dict["focus_position_y_m"] = fp[1]

            fs = fd.get("focus_size", {})
            params_dict["focus_fwhm_x_m"] = fs.get("fwhm_x", 0)
            params_dict["focus_fwhm_y_m"] = fs.get("fwhm_y", 0)

            params_dict["dx_focus_m"] = fd.get("dx_focus", 0)
            params_dict["dy_focus_m"] = fd.get("dy_focus", 0)

        # =========================================================
        # Save unified params file (with section separators)
        # =========================================================
        with open(frame_dir / "params.txt", "w") as f:
            f.write("# Pipeline Parameters\n")
            f.write(f"# Frame ID: {data.frame_id}\n")
            f.write("=" * 50 + "\n\n")

            # Wavefront fitting section
            f.write("# [Wavefront Fitting]\n")
            f.write("-" * 30 + "\n")
            for key in [
                "fit_x0_m",
                "fit_y0_m",
                "fit_Rx_m",
                "fit_Ry_m",
                "fit_A_rad",
                "virtual_pixel_size_x_m",
                "virtual_pixel_size_y_m",
                "wavelength_m",
            ]:
                if key in params_dict:
                    f.write(f"{key}: {params_dict[key]}\n")
            f.write("\n")

            # Calibration section
            f.write("# [Calibration]\n")
            f.write("-" * 30 + "\n")
            for key in [
                "cal_R_x_m",
                "cal_R_y_m",
                "cal_R_avg_m",
                "cal_delta_R_x_m",
                "cal_delta_R_y_m",
                "source_distance",
            ]:
                if key in params_dict:
                    f.write(f"{key}: {params_dict[key]}\n")
            f.write("\n")

            # ROI section
            f.write("# [ROI]\n")
            f.write("-" * 30 + "\n")
            for key in ["roi_center_x_m", "roi_center_y_m", "roi_radius"]:
                if key in params_dict:
                    f.write(f"{key}: {params_dict[key]}\n")
            f.write("\n")

            # Beam section
            f.write("# [Beam @ Detector]\n")
            f.write("-" * 30 + "\n")
            for key in [
                "beam_position_x_m",
                "beam_position_y_m",
                "beam_fwhm_x_m",
                "beam_fwhm_y_m",
            ]:
                if key in params_dict:
                    f.write(f"{key}: {params_dict[key]}\n")
            f.write("\n")

            # Focus section
            f.write("# [Focus]\n")
            f.write("-" * 30 + "\n")
            for key in [
                "focus_position_x_m",
                "focus_position_y_m",
                "focus_fwhm_x_m",
                "focus_fwhm_y_m",
                "dx_focus_m",
                "dy_focus_m",
            ]:
                if key in params_dict:
                    f.write(f"{key}: {params_dict[key]}\n")

    def stop(self):
        self._stop_event.set()
        self._queue.put(None)


# =============================================================================
# Main Runner
# =============================================================================


def run(
    fps: float = 10.0,
    duration: float = None,
    output_dir: Optional[Path] = None,
    queue_size: int = 5,
):
    """Run streaming pipeline."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "stream"

    print("=" * 50)
    print("XGI Pipeline - Streaming Mode")
    print("=" * 50)
    print(f"FPS: {fps} Hz")
    print(f"Duration: {'Infinite (Ctrl+C)' if duration is None else f'{duration}s'}")
    print(f"Output: {output_dir}")
    print("=" * 50)

    # Initialize
    params = get_params()
    image = np.array(Image.open("data/sample_exp.tif"))
    print(f"[Init] Image: {image.shape}")

    frame_queue = queue.Queue(maxsize=queue_size)

    # Create three components
    reader = DataReader(image, frame_queue, fps=fps, duration=duration)
    processor = DataProcessor(params)
    saver = DataSaver(output_dir)

    # Ctrl+C handler
    stop_flag = threading.Event()

    def signal_handler(sig, frame):
        print("\n[Stop] Shutting down...")
        stop_flag.set()
        reader.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Start threads
    reader.start()
    saver.start()

    # Main loop: Read -> Process -> Save
    processed = 0
    failed = 0

    print("[Running]")

    while not stop_flag.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
            if frame is None:
                break

            result = processor.process(frame)

            if result.success:
                saver.save(result)
                processed += 1
                print(f"[Frame {frame.frame_id:04d}] ✓")
            else:
                failed += 1

        except queue.Empty:
            continue

    # Cleanup
    reader.join(timeout=2.0)
    saver.stop()
    saver.join(timeout=5.0)

    print("=" * 50)
    print(
        f"Total: {reader.total_frames}, Dropped: {reader.dropped_frames}, Processed: {processed}, Saved: {saver.saved_count}"
    )
    print("=" * 50)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGI Streaming Runner")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    run(fps=args.fps, duration=args.duration, output_dir=output_dir)
