"""Unit tests for the interruptible service-call wait (cancel responsiveness)."""

import threading
import time
import unittest

from omni_slam_manager.cancel_wait import wait_with_cancel


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept = []

    def now_fn(self) -> float:
        return self.now

    def sleep_fn(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class WaitWithCancelTests(unittest.TestCase):
    def test_returns_true_when_done_immediately(self):
        clock = _FakeClock()
        self.assertTrue(
            wait_with_cancel(
                lambda: True,
                lambda: False,
                5.0,
                sleep=clock.sleep_fn,
                now=clock.now_fn,
            )
        )
        self.assertEqual(clock.slept, [])

    def test_returns_true_when_done_after_one_poll(self):
        clock = _FakeClock()
        states = {"done": False}

        def sleep_and_flip(seconds: float) -> None:
            clock.slept.append(seconds)
            clock.now += seconds
            states["done"] = True

        self.assertTrue(
            wait_with_cancel(
                lambda: states["done"],
                lambda: False,
                5.0,
                sleep=sleep_and_flip,
                now=clock.now_fn,
            )
        )
        self.assertEqual(clock.slept, [0.1])

    def test_returns_false_when_cancel_requested(self):
        clock = _FakeClock()
        flag = {"cancelled": False}

        def sleep_fn(seconds: float) -> None:
            clock.slept.append(seconds)
            clock.now += seconds
            if clock.now >= 0.25:
                flag["cancelled"] = True

        self.assertFalse(
            wait_with_cancel(
                lambda: False,
                lambda: flag["cancelled"],
                5.0,
                sleep=sleep_fn,
                now=clock.now_fn,
            )
        )

    def test_returns_false_on_timeout(self):
        clock = _FakeClock()
        self.assertFalse(
            wait_with_cancel(
                lambda: False,
                lambda: False,
                0.25,
                sleep=clock.sleep_fn,
                now=clock.now_fn,
            )
        )
        # no runaway past the deadline
        self.assertLessEqual(clock.now, 0.25)
        self.assertGreaterEqual(clock.slept, [0.1])

    def test_cancel_before_wait_returns_false_without_sleeping(self):
        clock = _FakeClock()
        self.assertFalse(
            wait_with_cancel(
                lambda: False,
                lambda: True,
                5.0,
                sleep=clock.sleep_fn,
                now=clock.now_fn,
            )
        )
        self.assertEqual(clock.slept, [])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            wait_with_cancel(
                lambda: False, lambda: False, 1.0, poll_sec=0.0
            )
        with self.assertRaises(ValueError):
            wait_with_cancel(lambda: False, lambda: False, -1.0)

    def test_real_cancel_is_honored_promptly(self):
        # A cancel flag flipped by a real thread must be honored well under
        # the 30 s service timeout this wait replaces.
        flag = {"cancelled": False}

        def flipper() -> None:
            time.sleep(0.15)
            flag["cancelled"] = True

        thread = threading.Thread(target=flipper)
        thread.start()
        started = time.monotonic()
        self.assertFalse(
            wait_with_cancel(
                lambda: False, lambda: flag["cancelled"], 30.0
            )
        )
        elapsed = time.monotonic() - started
        thread.join()
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()