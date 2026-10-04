"""Unit tests for render_grid.py's pure-numpy half: tiling, labels, and the coverage encoding.

These run WITHOUT Isaac. That is the point of them: the composition code is the part that can be
wrong in a way you only discover after an hour of GPU time has already been spent, and it is also
the only part that can be exercised on a box whose single GPU is busy. Importing render_grid must
therefore not launch AppLauncher -- if this file ever starts taking 30 s to import, the Isaac
imports have leaked back out of main().

Run either way:  python3 test_render_grid.py   |   python3 -m pytest test_render_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_grid import (  # noqa: E402
    GLYPH_H,
    GLYPH_W,
    best_simultaneous_window,
    compose_grid,
    coverage_runs,
    draw_text,
    grid_shape,
    outline_tile,
    pad_to_even,
    text_size,
)


def test_grid_shape_is_square_for_squares():
    assert grid_shape(1) == (1, 1)
    assert grid_shape(4) == (2, 2)
    assert grid_shape(9) == (3, 3)
    assert grid_shape(16) == (4, 4)
    # non-squares fill left to right, so cols >= rows and no row is entirely empty
    for n in range(1, 33):
        rows, cols = grid_shape(n)
        assert rows * cols >= n, n
        assert (rows - 1) * cols < n, n  # no wholly blank final row
        assert cols >= rows, n


def test_compose_grid_places_each_tile_in_its_own_cell():
    th, tw = 6, 8
    tiles = [np.full((th, tw, 3), v, dtype=np.uint8) for v in (10, 20, 30, 40)]
    out = compose_grid(tiles, 2, 2, th, tw)
    assert out.shape == (2 * th, 2 * tw, 3)
    assert out.dtype == np.uint8
    assert np.all(out[0:th, 0:tw] == 10)
    assert np.all(out[0:th, tw:2 * tw] == 20)
    assert np.all(out[th:2 * th, 0:tw] == 30)
    assert np.all(out[th:2 * th, tw:2 * tw] == 40)


def test_compose_grid_survives_missing_and_misshapen_tiles():
    """A warm-up render returns a zero-size buffer and a failed one returns None; neither may raise."""
    th, tw = 4, 4
    tiles = [
        np.full((th, tw, 3), 99, dtype=np.uint8),
        None,
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.full((th + 5, tw + 5, 4), 7, dtype=np.uint8),  # oversized, RGBA
    ]
    out = compose_grid(tiles, 2, 2, th, tw)
    assert np.all(out[0:th, 0:tw] == 99)
    assert np.all(out[0:th, tw:2 * tw] == 0)      # None -> black cell
    assert np.all(out[th:2 * th, 0:tw] == 0)      # empty buffer -> black cell
    assert np.all(out[th:2 * th, tw:2 * tw] == 7)  # cropped to the cell, alpha dropped


def test_compose_grid_pads_short_tile_lists_and_ignores_extras():
    th, tw = 3, 3
    out = compose_grid([np.full((th, tw, 3), 5, np.uint8)], 2, 2, th, tw)
    assert out.shape == (2 * th, 2 * tw, 3)
    assert out[th:, tw:].max() == 0
    # more tiles than cells must not overflow the buffer
    many = [np.full((th, tw, 3), 5, np.uint8)] * 9
    assert compose_grid(many, 2, 2, th, tw).shape == (2 * th, 2 * tw, 3)


def test_draw_text_writes_pixels_and_leaves_background_alone():
    img = np.zeros((40, 120, 3), dtype=np.uint8)
    draw_text(img, "E0", 4, 4, scale=2, color=(255, 255, 255))
    assert img.max() == 255
    w, h = text_size("E0", 2)
    assert (w, h) == (2 * (GLYPH_W + 1) * 2, GLYPH_H * 2)
    # nothing is written outside the text box
    assert img[4 + h:, :].max() == 0
    assert img[:, 4 + w:].max() == 0


def test_draw_text_background_plate_covers_the_label():
    img = np.full((40, 120, 3), 200, dtype=np.uint8)
    draw_text(img, "OK", 10, 10, scale=2, color=(0, 255, 0), bg=(16, 16, 20))
    # the plate replaced the bright background under the glyphs
    assert (img == 16).any()
    assert tuple(img[10, 10]) != (200, 200, 200)


def test_draw_text_clips_instead_of_raising():
    """A label near the edge degrades to a partial label; a caption never costs a frame."""
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    draw_text(img, "88888888", 6, 6, scale=3, color=(255, 0, 0))
    draw_text(img, "88", -5, -5, scale=3, color=(0, 0, 255))
    draw_text(img, "88", 100, 100, scale=3, color=(0, 255, 0))
    assert img.shape == (12, 12, 3)


def test_draw_text_skips_unknown_glyphs_without_shifting_the_rest():
    a = np.zeros((20, 80, 3), dtype=np.uint8)
    b = np.zeros((20, 80, 3), dtype=np.uint8)
    draw_text(a, "A~B", 0, 0, scale=2)
    draw_text(b, "A B", 0, 0, scale=2)
    # '~' has no glyph and ' ' is an all-blank glyph, so both render identically
    assert np.array_equal(a, b)


def test_draw_text_is_case_insensitive():
    a = np.zeros((20, 80, 3), dtype=np.uint8)
    b = np.zeros((20, 80, 3), dtype=np.uint8)
    draw_text(a, "ok", 0, 0, scale=2)
    draw_text(b, "OK", 0, 0, scale=2)
    assert np.array_equal(a, b)


def test_outline_tile_borders_only_its_own_cell():
    th, tw = 10, 12
    frame = np.zeros((2 * th, 2 * tw, 3), dtype=np.uint8)
    outline_tile(frame, 3, 2, th, tw, (0, 255, 0), thickness=2)  # bottom-right cell
    cell = frame[th:, tw:]
    assert np.all(cell[0:2, :] == (0, 255, 0))
    assert np.all(cell[-2:, :] == (0, 255, 0))
    assert np.all(cell[:, 0:2] == (0, 255, 0))
    assert np.all(cell[:, -2:] == (0, 255, 0))
    assert cell[2:-2, 2:-2].max() == 0            # interior untouched
    assert frame[0:th, 0:tw].max() == 0           # other cells untouched
    assert frame[0:th, tw:].max() == 0


def test_outline_tile_thickness_cannot_swallow_a_tiny_tile():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    outline_tile(frame, 0, 1, 4, 4, (255, 0, 0), thickness=99)
    assert frame[:, :, 0].min() == 0 or frame.shape == (4, 4, 3)  # no exception, no overflow


def test_pad_to_even_only_grows_odd_dimensions():
    even = np.ones((4, 6, 3), np.uint8)
    assert pad_to_even(even) is even
    odd = np.ones((5, 7, 3), np.uint8)
    out = pad_to_even(odd)
    assert out.shape == (6, 8, 3)
    assert np.all(out[:5, :7] == 1)
    assert out[5, :].max() == 0 and out[:, 7].max() == 0


def _eps(*spans):
    return [{"frame_start": s, "frame_end": e, "held_at_end": h, "ever_during": v}
            for s, e, h, v in spans]


def test_coverage_runs_encodes_overlapping_episodes_exactly():
    # env0 succeeds over frames 0-4, env1 over 3-9. The overlap 3-4 is the window a cutter wants.
    by_env = [_eps((0, 4, True, True), (5, 9, False, False)),
              _eps((0, 2, False, False), (3, 9, True, True))]
    runs = coverage_runs(by_env, 2, 10)
    assert [r[0] for r in runs] == [0, 3, 5]
    assert [r[1] for r in runs] == [2, 4, 9]
    assert [r[3] for r in runs] == [0b01, 0b11, 0b10]   # held_mask
    assert [r[4] for r in runs] == [0b01, 0b11, 0b10]   # ever_mask
    assert all(r[2] == 0b11 for r in runs)              # both envs always mid-episode
    # the encoding is lossless: expanding the runs reproduces a per-frame answer
    per_frame = {}
    for s, e, active, held, everm in runs:
        for f in range(s, e + 1):
            per_frame[f] = (active, held, everm)
    assert len(per_frame) == 10
    assert per_frame[3] == (0b11, 0b11, 0b11)
    assert per_frame[7] == (0b11, 0b10, 0b10)


def test_coverage_runs_is_compact_not_per_frame():
    """The whole reason for run-length: 4 envs x 12 episodes over 9600 frames stays tiny."""
    n_envs, n_eps, span = 4, 12, 200
    by_env = [_eps(*[(k * span, (k + 1) * span - 1, k % 3 == 0, k % 2 == 0) for k in range(n_eps)])
              for _ in range(n_envs)]
    runs = coverage_runs(by_env, n_envs, n_eps * span)
    assert len(runs) <= 2 * n_envs * n_eps
    assert len(runs) < n_eps * span / 10  # orders of magnitude below one record per frame


def test_coverage_runs_drops_episodes_shorter_than_one_recorded_frame():
    """With --render_every > 1 an episode can end between frames; frame_end < frame_start."""
    by_env = [_eps((5, 4, True, True), (5, 8, False, False))]
    runs = coverage_runs(by_env, 1, 10)
    covered = {f for s, e, a, _h, _v in runs if a for f in range(s, e + 1)}
    assert covered == {5, 6, 7, 8}


def test_coverage_runs_clamps_to_the_recorded_frame_count():
    by_env = [_eps((0, 999, True, True))]
    runs = coverage_runs(by_env, 1, 5)
    assert runs[-1][1] == 4
    assert coverage_runs(by_env, 1, 0) == []


def test_coverage_runs_rejects_more_envs_than_the_mask_holds():
    try:
        coverage_runs([[] for _ in range(64)], 64, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 64 envs")


def test_best_simultaneous_window_finds_the_overlap():
    by_env = [_eps((0, 4, True, True), (5, 9, False, False)),
              _eps((0, 2, False, False), (3, 9, True, True))]
    runs = coverage_runs(by_env, 2, 10)
    best = best_simultaneous_window(runs, mask_field=3)
    assert best["n_envs"] == 2
    assert (best["frame_start"], best["frame_end"]) == (3, 4)
    assert best["mask"] == 0b11


def test_best_simultaneous_window_prefers_the_longer_of_equal_counts():
    by_env = [_eps((0, 1, True, True), (2, 3, False, False), (4, 9, True, True))]
    runs = coverage_runs(by_env, 1, 10)
    best = best_simultaneous_window(runs, mask_field=3)
    assert (best["frame_start"], best["frame_end"]) == (4, 9)


def test_best_simultaneous_window_reports_zero_when_nothing_succeeds():
    runs = coverage_runs([_eps((0, 9, False, False))], 1, 10)
    assert best_simultaneous_window(runs, mask_field=3)["n_envs"] == 0


def test_full_frame_pipeline_matches_the_ffmpeg_byte_count():
    """End-to-end shape check: what the loop pipes must be exactly grid_w*grid_h*3 bytes."""
    n_envs, th, tw = 4, 360, 640
    rows, cols = grid_shape(n_envs)
    tiles = [np.random.randint(0, 255, (th, tw, 3), dtype=np.uint8) for _ in range(n_envs)]
    frame = compose_grid(tiles, rows, cols, th, tw)
    for i in range(n_envs):
        outline_tile(frame, i, cols, th, tw, (60, 220, 90))
        r, c = divmod(i, cols)
        draw_text(frame, f"E{i}  12MM OK", c * tw + 6, r * th + 6, scale=2, bg=(16, 16, 20))
    frame = pad_to_even(frame)
    assert frame.shape == (rows * th, cols * tw, 3) == (720, 1280, 3)
    assert len(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()) == 1280 * 720 * 3


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
