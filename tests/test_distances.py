import numpy as np

from pybeepbeep.ranging import calculate_distances, index_distances


def test_index_distances():
    # a real distance array would be mirrored over the identity diagonal, but only the top half matters
    distances = np.array([[0, 3, 4, 5],
                          [0, 0, 6, 7],
                          [0, 0, 0, 7],
                          [0, 0, 0, 0]])

    schedule = [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]

    expected_indexes = {
        "1": {
            "1": 0,
            "2": distances[0][1],
            "3": distances[0][2],
            "4": distances[0][3]
        },
        "2": {
            "1": distances[0][1],
            "2": 0,
            "3": distances[1][2],
            "4": distances[1][3]
        },
        "3": {
            "1": distances[0][2],
            "2": distances[1][2],
            "3": 0,
            "4": distances[2][3]
        },
        "4": {
            "1": distances[0][3],
            "2": distances[1][3],
            "3": distances[2][3],
            "4": 0
        }
    }

    indexes = index_distances(distances, schedule)

    assert expected_indexes == indexes


def test_calc_distances_no_k():
    d_1_1 = 0
    d_2_2 = 0
    d_3_3 = 0
    d_4_4 = 0

    d_1_2 = 300
    d_2_1 = d_1_2 + 300

    d_1_3 = 3000
    d_3_1 = d_1_3 + 500

    d_1_4 = 30000
    d_4_1 = d_1_4 + 800

    d_2_3 = 300000
    d_3_2 = d_2_3 - 300

    d_2_4 = 3000000
    d_4_2 = d_2_4 - 500

    d_3_4 = 30000000
    d_4_3 = d_3_4 - 800

    f = 343 / (2 * 44100)

    d_300 = f * 300
    d_500 = f * 500
    d_800 = f * 800

    d = np.array([[d_1_1, d_1_2, d_1_3, d_1_4],
                  [d_2_1, d_2_2, d_2_3, d_2_4],
                  [d_3_1, d_3_2, d_3_3, d_3_4],
                  [d_4_1, d_4_2, d_4_3, d_4_4]])

    distances = calculate_distances(deltas=d,
                                    sampling_freq_hz=44100)

    expected_distances = np.array([[    0, d_300, d_500, d_800],  # noqa: E201
                                   [d_300,     0, d_300, d_500],  # noqa: E241
                                   [d_500, d_300,     0, d_800],  # noqa: E241
                                   [d_800, d_500, d_800,     0]])  # noqa: E241

    assert np.array_equal(distances, expected_distances)


def test_calc_distances_with_ks():
    d_1_1 = 10
    d_2_2 = 20
    d_3_3 = 30
    d_4_4 = 40

    d_1_2 = 300
    d_2_1 = d_1_2 + 300

    d_1_3 = 3000
    d_3_1 = d_1_3 + 500

    d_1_4 = 30000
    d_4_1 = d_1_4 + 800

    d_2_3 = 300000
    d_3_2 = d_2_3 - 300

    d_2_4 = 3000000
    d_4_2 = d_2_4 - 500

    d_3_4 = 30000000
    d_4_3 = d_3_4 - 800

    f = 343 / (2 * 44100)

    d = np.array([[d_1_1, d_1_2, d_1_3, d_1_4],
                  [d_2_1, d_2_2, d_2_3, d_2_4],
                  [d_3_1, d_3_2, d_3_3, d_3_4],
                  [d_4_1, d_4_2, d_4_3, d_4_4]])

    distances = calculate_distances(deltas=d,
                                    sampling_freq_hz=44100)

    d11 = 2 * d_1_1 * f
    d12 = f * (300 + d_1_1 + d_2_2)
    d13 = f * (500 + d_1_1 + d_3_3)
    d14 = f * (800 + d_1_1 + d_4_4)

    d22 = 2 * d_2_2 * f
    d23 = f * (300 + d_2_2 + d_3_3)
    d24 = f * (500 + d_2_2 + d_4_4)

    d33 = 2 * d_3_3 * f
    d34 = f * (800 + d_3_3 + d_4_4)

    d44 = 2 * d_4_4 * f

    expected_distances = np.array([[d11, d12, d13, d14],
                                   [d12, d22, d23, d24],
                                   [d13, d23, d33, d34],
                                   [d14, d24, d34, d44]])

    assert np.array_equal(distances, expected_distances)


def test_calc_distances_with_c():
    d_1_1 = 10
    d_2_2 = 20
    d_3_3 = 30
    d_4_4 = 40

    d_1_2 = 300
    d_2_1 = d_1_2 + 300

    d_1_3 = 3000
    d_3_1 = d_1_3 + 500

    d_1_4 = 30000
    d_4_1 = d_1_4 + 800

    d_2_3 = 300000
    d_3_2 = d_2_3 - 300

    d_2_4 = 3000000
    d_4_2 = d_2_4 - 500

    d_3_4 = 30000000
    d_4_3 = d_3_4 - 800

    f = 100 / (2 * 44100)

    d = np.array([[d_1_1, d_1_2, d_1_3, d_1_4],
                  [d_2_1, d_2_2, d_2_3, d_2_4],
                  [d_3_1, d_3_2, d_3_3, d_3_4],
                  [d_4_1, d_4_2, d_4_3, d_4_4]])

    distances = calculate_distances(deltas=d,
                                    sampling_freq_hz=44100,
                                    c=100)

    d11 = 2 * d_1_1 * f
    d12 = f * (300 + d_1_1 + d_2_2)
    d13 = f * (500 + d_1_1 + d_3_3)
    d14 = f * (800 + d_1_1 + d_4_4)

    d22 = 2 * d_2_2 * f
    d23 = f * (300 + d_2_2 + d_3_3)
    d24 = f * (500 + d_2_2 + d_4_4)

    d33 = 2 * d_3_3 * f
    d34 = f * (800 + d_3_3 + d_4_4)

    d44 = 2 * d_4_4 * f

    expected_distances = np.array([[d11, d12, d13, d14],
                                   [d12, d22, d23, d24],
                                   [d13, d23, d33, d34],
                                   [d14, d24, d34, d44]])

    assert np.array_equal(distances, expected_distances)
