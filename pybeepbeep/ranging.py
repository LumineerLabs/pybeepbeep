import math
import numpy as np
# import matplotlib.pyplot as plt

from typing import Callable, List, Dict

from scipy.signal import correlate, hilbert, find_peaks
from librosa.core import tone, time_to_samples


# set fft window size
# this corresponds to a resolution of about 2% of the sampling frequency
_fft_width = 512


def _get_window_size_ms(duration_ms: float):
    return 20 * duration_ms


def get_minimum_channel_width(sampling_freq_hz: float):
    resolution = sampling_freq_hz / _fft_width
    return 10 * resolution


def _find_beep_in_window(samples: np.ndarray,
                         sampling_freq_hz: float,
                         target_signal_freq_hz: float,
                         duration_ms: float) -> int:
    if target_signal_freq_hz >= sampling_freq_hz / 2:
        raise Exception(
            "Sampling frequency must be > 2x the target frequency. See https://en.wikipedia.org/wiki/Nyquist_rate"
        )

    # generate target signal
    signal = tone(target_signal_freq_hz, sampling_freq_hz, duration=duration_ms/1000.0)

    # find onset, this differs from the description in the paper which uses a sharpness and peak finding algorithm
    correlation = correlate(samples, signal, mode='valid', method='fft')
    envelope = np.abs(hilbert(correlation))
    max_correlation = np.max(correlation)
    peaks, _ = find_peaks(envelope)
    filtered_peaks = peaks[(peaks < len(samples)).nonzero()]
    peaks = peaks[(envelope[filtered_peaks] > .85 * max_correlation).nonzero()]

    ratio = (np.max(signal) / np.max(correlation))

    correlation *= ratio
    envelope *= ratio

    if len(peaks) == 0:
        # if not found, use None
        return None
    else:
        return peaks[0]


def _calculate_windows_for_schedule(sampling_freq_hz: float,
                                    schedule: [{}]) -> [(int, int)]:
    if len(schedule) == 0:
        return None

    window_duration_s = _get_window_size_ms(schedule[0]["duration_ms"]) / 1000.0
    half_window_s = window_duration_s / 2

    return [
        (
            time_to_samples(entry["time_s"] - half_window_s, sampling_freq_hz),
            time_to_samples(entry["time_s"] + half_window_s, sampling_freq_hz)
        )
        for entry in schedule
    ]


def find_deltas(samples: np.ndarray,
                sampling_freq_hz: float,
                schedule: [{}],
                self_id: str) -> [float]:
    windows = _calculate_windows_for_schedule(sampling_freq_hz=sampling_freq_hz,
                                              schedule=schedule)
    onsets = np.zeros(len(schedule))
    self_n = 0

    for i, window in enumerate(windows):
        n_onset = _find_beep_in_window(samples=samples[window[0]:window[1]],
                                       sampling_freq_hz=sampling_freq_hz,
                                       target_signal_freq_hz=schedule[i]["target_hz"],
                                       duration_ms=schedule[i]["duration_ms"])

        if n_onset is None:
            onsets[i] = math.inf
        else:
            n_onset += window[0]
            onsets[i] = float(n_onset)

        if schedule[i]["id"] == self_id:
            self_n = n_onset

    return np.absolute(onsets - self_n)


def single_tone_scheduler(nodes: [str],
                          target_hz: float,
                          duration_ms: float):
    window = _get_window_size_ms(duration_ms) / 1000.0

    return [{"id": node, "target_hz": target_hz, "duration_ms": duration_ms, "time_s": (i * window) + window}
            for i, node in enumerate(nodes)]


def band_scheduler(nodes: [str],
                   channels: [float],
                   duration_ms: float):
    n_windows = math.ceil(len(nodes) / float(len(channels)))

    schedule = []

    for i in range(len(channels)):
        index = i * n_windows
        channel_schedule = single_tone_scheduler(nodes=nodes[index:index + n_windows],
                                                 target_hz=channels[i],
                                                 duration_ms=duration_ms)
        schedule.extend(channel_schedule)

    return schedule


def generate_schedule(nodes: [str],
                      schedule_strategy: Callable[[List[str], List[float], float], List[Dict]] = single_tone_scheduler,
                      scheduler_kwargs: {} = None) -> [{}]:
    if scheduler_kwargs is None:
        scheduler_kwargs = {"target_hz": 6000, "duration_ms": 50}
    return schedule_strategy(nodes, **scheduler_kwargs)


def calculate_distances(deltas: np.ndarray, sampling_freq_hz: float, c: float = 343) -> np.ndarray:
    """
    If the caller wants to account for the distance between the speaker and microphone on the node, the k factors
    should be converted to a sample count and placed in the diagonal of the deltas matrix (d1,1, d2,2, etc.).

    d = [d1,1 d1,2 d1,3 d1,4]
        [d2,1 d2,2 d2,3 d2,4]
        [d3,1 d3,2 d3,3 d3,4]
        [d4,1 d4,2 d4,3 d4,4]

    distance = (c / 2fs)(|d1,2 - d2,1| + d1,1 + d2,2)

    (c/2fs)|d-dT|+

    [0         |d12-d21| |d13-d31| |d14-d41|]  [0       d11+d22 d11+d33 d11+d44]
    [|d21-d12| 0         |d23-d32| |d24-d42|]  [d22+d11 0       d22+d33 d22+d44]
    [ ...         ...    0         |d34-d43|]  [                0       d33+d44]
    [ ...         ...       ...    0        ]  [                        0      ]

    [d11 d11 d11 d11] [d11 d22 d33 d44]
    [d22 d22 d22 d22] [d11 d22 d33 d44]
    [d33 d33 d33 d33] [d11 d22 d33 d44]
    [d44 d44 d44 d44] [d11 d22 d33 d44]
    """
    conversion_factor = c / (2 * sampling_freq_hz)

    deltas_t = deltas.T

    k1 = deltas * np.eye(deltas.shape[0]) @ np.ones(deltas.shape)
    k2 = k1.T
    k = k1 + k2

    return conversion_factor * (np.abs(deltas - deltas_t) + k)


def index_distances(distances: np.ndarray, schedule: List[Dict]) -> Dict[str, Dict]:
    indexed_distances = {}
    for i in range(len(schedule)):
        i_id = schedule[i]["id"]
        for j in range(i, len(schedule)):
            j_id = schedule[j]["id"]
            distance = distances[i][j]

            if i_id not in indexed_distances.keys():
                indexed_distances[i_id] = {}

            if j_id not in indexed_distances.keys():
                indexed_distances[j_id] = {}

            indexed_distances[i_id][j_id] = distance
            indexed_distances[j_id][i_id] = distance

    return indexed_distances
