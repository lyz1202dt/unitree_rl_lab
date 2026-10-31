"""Offline temporal contract checks; no simulator or robot is started."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

MODULE = Path(__file__).resolve().parents[1] / "source/unitree_rl_lab/unitree_rl_lab/utils/rs03_network.py"
spec = importlib.util.spec_from_file_location("rs03_network", MODULE)
rs03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs03)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "actuator.pt"
        model = torch.nn.Linear(6, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 10., 100., 1000., 10000., 100000.]]))
        torch.jit.script(model).save(str(self.path))

    def tearDown(self):
        self.temporary.cleanup()

    def test_history_matches_csv_features_and_has_no_extra_delay(self):
        model = rs03.HistoryTorqueModel(self.path, 2, 12, "cpu")
        for step in range(1, 4):
            errors = torch.arange(24).reshape(2, 12).float() + step
            velocity = errors * 2
            output = model.step(errors, velocity)
        expected = model.network(torch.stack((errors, errors - 1, errors - 2,
                                              velocity, velocity - 2, velocity - 4), dim=-1).reshape(-1, 6))
        torch.testing.assert_close(output, expected.reshape(2, 12))

    def test_reset_only_changes_selected_environment_and_repeats_first_state(self):
        model = rs03.HistoryTorqueModel(self.path, 2, 12, "cpu")
        model.step(torch.ones(2, 12), torch.full((2, 12), 2.))
        model.step(torch.full((2, 12), 3.), torch.full((2, 12), 4.))
        model.reset([0])
        model.step(torch.full((2, 12), 5.), torch.full((2, 12), 6.))
        self.assertEqual(model.errors[0, 0].tolist(), [5., 5., 5.])
        self.assertEqual(model.errors[1, 0].tolist(), [5., 3., 1.])
        self.assertEqual(model.velocities[0, 0].tolist(), [6., 6., 6.])

    def test_contract_rejects_wrong_physics_timing_or_input_order(self):
        contract = {"inputs": rs03.INPUTS, "position_error": "clipped_target_q - feedback_q",
                    "label": "joint_torque_estimate_at_next_host_sample", "label_offset_rows": 1,
                    "state": "RLFSMStateRLLocomotion", "joint_order": rs03.JOINTS, "nominal_dt_s": .005}
        self.path.with_suffix(".json").write_text(json.dumps(contract))
        rs03.read_contract(self.path, None, .005)
        with self.assertRaisesRegex(ValueError, "物理步长"):
            rs03.read_contract(self.path, None, .02)
        contract["inputs"] = list(reversed(rs03.INPUTS))
        self.path.with_suffix(".json").write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, "inputs"):
            rs03.read_contract(self.path, None, .005)


if __name__ == "__main__":
    unittest.main()
