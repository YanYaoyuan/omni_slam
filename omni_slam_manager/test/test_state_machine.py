import unittest

from omni_slam_manager.state_machine import (
    InvalidTransition,
    SlamMode,
    SlamState,
    SlamStateMachine,
)


class SlamStateMachineTests(unittest.TestCase):
    def test_mapping_save_and_stop_flow(self):
        machine = SlamStateMachine()
        machine.transition(SlamState.STARTING_MAPPING)
        machine.transition(SlamState.MAPPING)
        machine.transition(SlamState.SAVING)
        snapshot = machine.transition(SlamState.MAP_READY)
        self.assertEqual(snapshot.mode, SlamMode.MAPPING)
        self.assertEqual(snapshot.state, SlamState.MAP_READY)
        machine.transition(SlamState.STOPPING)
        self.assertEqual(machine.transition(SlamState.STOPPED).mode, SlamMode.STOPPED)

    def test_localization_quality_flow(self):
        machine = SlamStateMachine()
        machine.transition(SlamState.STARTING_LOCALIZATION)
        machine.transition(SlamState.RELOCALIZING)
        machine.transition(SlamState.LOCALIZED)
        machine.transition(SlamState.DEGRADED)
        machine.transition(SlamState.LOST)
        snapshot = machine.transition(SlamState.RELOCALIZING)
        self.assertEqual(snapshot.mode, SlamMode.LOCALIZATION)

    def test_invalid_transition_is_rejected_without_state_change(self):
        machine = SlamStateMachine()
        before = machine.snapshot
        with self.assertRaises(InvalidTransition):
            machine.transition(SlamState.LOCALIZED)
        self.assertEqual(machine.snapshot, before)

    def test_error_requires_reason_and_can_be_stopped(self):
        machine = SlamStateMachine()
        machine.transition(SlamState.STARTING_MAPPING)
        with self.assertRaises(ValueError):
            machine.fail(0, "")
        failed = machine.fail(1001, "FAST-LIO exited")
        self.assertEqual(failed.state, SlamState.ERROR)
        self.assertEqual(failed.reason_code, 1001)
        machine.transition(SlamState.STOPPING)
        self.assertEqual(machine.transition(SlamState.STOPPED).state, SlamState.STOPPED)

    def test_sequence_only_advances_on_a_change(self):
        machine = SlamStateMachine()
        initial = machine.snapshot.sequence
        machine.transition(SlamState.STOPPED)
        self.assertEqual(machine.snapshot.sequence, initial)
        machine.transition(SlamState.STARTING_MAPPING)
        self.assertEqual(machine.snapshot.sequence, initial + 1)


if __name__ == "__main__":
    unittest.main()
