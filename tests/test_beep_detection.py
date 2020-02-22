import numpy as np

from librosa.core import time_to_samples, samples_to_time
from librosa.core import tone as create_tone

from scipy.signal.windows import hamming

from pybeepbeep.ranging import _find_beep_in_window, band_scheduler, generate_schedule, find_deltas, \
                               _calculate_windows_for_schedule, _get_window_size_ms


onset_accuracy_threshold = .2
delta_accuracy_threshold = time_to_samples(.01/343)


def create_clip(tones: [{}]=[],
                duration_s: float=5.0,
                sampling_rate_hz: float=44100.0,
                background_freq_hz: float=None) -> np.ndarray:
    n_samples = time_to_samples(duration_s, sr=44100)
    signal = np.zeros(n_samples)
    if background_freq_hz is not None:
        signal = create_tone(frequency=background_freq_hz, sr=sampling_rate_hz, duration=duration_s)

    for tone in tones:
        tone_signal = create_tone(frequency=tone["freq_hz"], sr=sampling_rate_hz, duration=tone["duration_s"])
        window = hamming(len(tone_signal))
        tone_signal *= window
        start_sample = time_to_samples(tone["start_s"], sr=44100)
        pre_buf = np.zeros(start_sample)
        tone_signal = np.concatenate((pre_buf, tone_signal))
        post_buf = np.zeros(n_samples - len(tone_signal))
        tone_signal = np.concatenate((tone_signal, post_buf))
        signal += tone_signal

    return signal


def test_detect_beep_none():
    clip = create_clip()
    found = _find_beep_in_window(samples=clip,
                                 sampling_freq_hz=44100.0,
                                 target_signal_freq_hz=8000.0,
                                 duration_ms=50.0)
    assert found is None


def test_detect_beep_no_background():
    clip = create_clip(
        [
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 2.5}
        ]
    )

    found = _find_beep_in_window(samples=clip,
                                 sampling_freq_hz=44100.0,
                                 target_signal_freq_hz=8000.0,
                                 duration_ms=50.0)

    assert found is not None
    assert abs(samples_to_time(found, 44100) - 2.5) < onset_accuracy_threshold


def test_detect_beep_with_background():
    clip = create_clip(
        [
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 2.5}
        ],
        background_freq_hz=1000.0
    )

    found = _find_beep_in_window(samples=clip,
                                 sampling_freq_hz=44100.0,
                                 target_signal_freq_hz=8000.0,
                                 duration_ms=50.0)

    assert found is not None
    assert abs(samples_to_time(found, 44100) - 2.5) < onset_accuracy_threshold


def test_detect_beep_multiple_without_overlap():
    clip = create_clip(
        [
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 3.1},
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 2.5}
        ],
        background_freq_hz=1000.0
    )

    found = _find_beep_in_window(samples=clip,
                                 sampling_freq_hz=44100.0,
                                 target_signal_freq_hz=8000.0,
                                 duration_ms=50.0)

    assert found is not None
    assert abs(samples_to_time(found, 44100) - 2.5) < onset_accuracy_threshold


def test_detect_beep_multiple_with_overlap():
    clip = create_clip(
        [
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 2.53},
            {"freq_hz": 8000.0, "duration_s": .05, "start_s": 2.5}
        ],
        background_freq_hz=1000.0
    )

    found = _find_beep_in_window(samples=clip,
                                 sampling_freq_hz=44100.0,
                                 target_signal_freq_hz=8000.0,
                                 duration_ms=50.0)

    assert found is not None
    assert abs(samples_to_time(found, 44100) - 2.5) < onset_accuracy_threshold


def test_calc_windows_single():
    f_sampling = 44100.0

    schedule = [
        {"time_s": 2.5, "duration_ms": 50},
    ]

    expected_windows = [
        (time_to_samples(2, f_sampling), time_to_samples(3, f_sampling))
    ]

    windows = _calculate_windows_for_schedule(sampling_freq_hz=f_sampling, schedule=schedule)

    assert expected_windows == windows


def test_calc_windows_multiple():
    f_sampling = 44100.0

    schedule = [
        {"time_s": 2.5, "duration_ms": 50},
        {"time_s": 3.5, "duration_ms": 50},
        {"time_s": 4.5, "duration_ms": 50},
    ]

    expected_windows = [
        (time_to_samples(2, f_sampling), time_to_samples(3, f_sampling)),
        (time_to_samples(3, f_sampling), time_to_samples(4, f_sampling)),
        (time_to_samples(4, f_sampling), time_to_samples(5, f_sampling))
    ]

    windows = _calculate_windows_for_schedule(sampling_freq_hz=f_sampling, schedule=schedule)

    assert expected_windows == windows


def test_calc_windows_multiple_overlapping():
    f_sampling = 44100.0

    schedule = [
        {"time_s": 2.5, "duration_ms": 50},
        {"time_s": 3, "duration_ms": 50},
        {"time_s": 3, "duration_ms": 50},
    ]

    expected_windows = [
        (time_to_samples(2, f_sampling), time_to_samples(3, f_sampling)),
        (time_to_samples(2.5, f_sampling), time_to_samples(3.5, f_sampling)),
        (time_to_samples(2.5, f_sampling), time_to_samples(3.5, f_sampling))
    ]

    windows = _calculate_windows_for_schedule(sampling_freq_hz=f_sampling, schedule=schedule)

    assert expected_windows == windows


def test_find_deltas_two_nodes():
    f_sampling = 44100.0
    nodes = ['1', '2']
    target_hz = 1000.0
    duration_ms = 1.0
    window = _get_window_size_ms(1.0)

    schedule = generate_schedule(nodes=nodes,
                                 scheduler_kwargs={
                                    "target_hz": target_hz,
                                    "duration_ms": duration_ms
                                 })

    tones = [{"freq_hz": entry["target_hz"], "duration_s": entry["duration_ms"] / 1000.0, "start_s": entry["time_s"]}
             for entry in schedule]

    clip = create_clip(tones=tones, duration_s=2.0, sampling_rate_hz=f_sampling)

    deltas = find_deltas(samples=clip, sampling_freq_hz=f_sampling, schedule=schedule, self_id='1')

    assert np.array_equal(deltas, [0, time_to_samples(window/1000, sr=f_sampling)])


def test_find_deltas_multi_tone():
    f_sampling = 44100.0
    nodes = ['1', '2', '3', '4']
    target_hz = [1000.0, 2000.0]
    duration_ms = 1.0
    window = _get_window_size_ms(1.0)

    schedule = generate_schedule(nodes=nodes,
                                 schedule_strategy=band_scheduler,
                                 scheduler_kwargs={
                                    "channels": target_hz,
                                    "duration_ms": duration_ms
                                 })

    tones = [{"freq_hz": entry["target_hz"], "duration_s": entry["duration_ms"] / 1000.0, "start_s": entry["time_s"]}
             for entry in schedule]

    clip = create_clip(tones=tones, duration_s=2.0, sampling_rate_hz=f_sampling)

    deltas = find_deltas(samples=clip, sampling_freq_hz=f_sampling, schedule=schedule, self_id='1')

    assert np.array_equal(deltas, [0, time_to_samples(window/1000, sr=f_sampling), 0, time_to_samples(window/1000, sr=f_sampling)])
