from pybeepbeep.ranging import single_tone_scheduler, band_scheduler, generate_schedule


def test_single_tone_scheduler_single_node():
    nodes = ['1']
    target_hz = 1000.0
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        }
    ]

    generated_schedule = single_tone_scheduler(nodes=nodes,
                                               target_hz=target_hz,
                                               duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_single_tone_scheduler_multiple_nodes():
    nodes = ['1', '2', '3']
    target_hz = 1000.0
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + window,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + 2 * window,
            "id": nodes[2]
        }
    ]

    generated_schedule = single_tone_scheduler(nodes=nodes,
                                               target_hz=target_hz,
                                               duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_band_scheduler_single_channel_single_node():
    nodes = ['1']
    target_hz = [1000.0]
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        }
    ]

    generated_schedule = band_scheduler(nodes=nodes,
                                        channels=target_hz,
                                        duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_band_scheduler_single_channel_multiple_nodes():
    nodes = ['1', '2', '3']
    target_hz = [1000.0]
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start + window,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start + 2 * window,
            "id": nodes[2]
        }
    ]

    generated_schedule = band_scheduler(nodes=nodes,
                                        channels=target_hz,
                                        duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_band_scheduler_multiple_channel_single_node():
    nodes = ['1']
    target_hz = [1000.0, 2000.0, 3000.0]
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        }
    ]

    generated_schedule = band_scheduler(nodes=nodes,
                                        channels=target_hz,
                                        duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_band_scheduler_multiple_channel_multiple_nodes():
    nodes = ['1', '2', '3']
    target_hz = [1000.0, 2000.0, 3000.0]
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz[1],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz[2],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[2]
        }
    ]

    generated_schedule = band_scheduler(nodes=nodes,
                                        channels=target_hz,
                                        duration_ms=duration_ms)

    assert generated_schedule == expected_schedule


def test_generate_schedule_defaults():
    nodes = ['1', '2', '3']
    target_hz = 6000.0
    duration_ms = 50.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + window,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + 2 * window,
            "id": nodes[2]
        }
    ]

    generated_schedule = generate_schedule(nodes=nodes)

    assert generated_schedule == expected_schedule



def test_generate_schedule_band_scheduler_multiple_channel_multiple_nodes():
    nodes = ['1', '2', '3']
    target_hz = [1000.0, 2000.0, 3000.0]
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz[0],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz[1],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz[2],
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[2]
        }
    ]

    generated_schedule = generate_schedule(nodes=nodes,
                                           schedule_strategy=band_scheduler,
                                           scheduler_kwargs={
                                               "channels": target_hz,
                                               "duration_ms": duration_ms
                                           })

    assert generated_schedule == expected_schedule


def test_generate_schedule_single_tone_scheduler_multiple_nodes():
    nodes = ['1', '2', '3']
    target_hz = 1000.0
    duration_ms = 1.0
    window = 20.0 * duration_ms / 1000.0
    start = window

    expected_schedule = [
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start,
            "id": nodes[0]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + window,
            "id": nodes[1]
        },
        {
            "target_hz": target_hz,
            "duration_ms": duration_ms,
            "time_s": start + 2 * window,
            "id": nodes[2]
        }
    ]

    generated_schedule = generate_schedule(nodes=nodes,
                                           schedule_strategy=single_tone_scheduler,
                                           scheduler_kwargs={
                                               "target_hz": target_hz,
                                               "duration_ms": duration_ms
                                           })

    assert generated_schedule == expected_schedule
