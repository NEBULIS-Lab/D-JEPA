import importlib.util
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import torch


class ReleaseTests(unittest.TestCase):
    def test_portable_package_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('djepa'))

    def test_ordinal_ties_follow_literal_candidate_id(self):
        from djepa.inference import ordinal_ranks
        ranks = ordinal_ranks(np.array([[2., 1., 1.]]), np.array([[7, 9, 2]]))
        np.testing.assert_array_equal(ranks, [[1., .5, 0.]])

    def test_gate_modes_are_distinct_and_strict(self):
        from djepa.inference import gated_positions
        base = np.array([[0., .2]])
        refined = np.array([[.1, .05]])
        ids = np.array([[1, 2]])
        self.assertEqual(gated_positions(base, refined, ids, .01, 'refined_gap')[0], 1)
        self.assertEqual(gated_positions(base, refined, ids, .01, 'base_minus_refined')[0], 0)
        self.assertEqual(gated_positions(base, refined, ids, .05, 'refined_gap')[0], 0)

    def test_duplicate_ids_rejected(self):
        from djepa.inference import ordinal_ranks
        with self.assertRaises(ValueError):
            ordinal_ranks(np.array([[0., 1.]]), np.array([[4, 4]]))

    def test_zero_head_preserves_base_and_is_permutation_equivariant(self):
        from djepa.ordinal import DecisionAligner
        torch.manual_seed(3)
        model = DecisionAligner(3).eval()
        features, base = torch.randn(2, 7, 3), torch.rand(2, 7)
        scores, delta = model(features, base)
        torch.testing.assert_close(scores, base, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(delta)), 0)
        torch.nn.init.normal_(model.correction_up.weight)
        order = torch.tensor([6, 2, 0, 4, 3, 1, 5])
        out = model(features, base)[0]
        permuted = model(features[:, order], base[:, order])[0]
        torch.testing.assert_close(permuted, out[:, order], atol=1e-6, rtol=1e-5)

    def test_actual_training_step_updates_head(self):
        from djepa.ordinal import DecisionAligner, decision_loss
        torch.manual_seed(5)
        model = DecisionAligner(3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        features, base = torch.randn(2, 7, 3), torch.rand(2, 7)
        labels = torch.tensor([[1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0]])
        scores, delta = model(features, base)
        loss = decision_loss(scores, delta, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertGreater(float(model.correction_up.weight.detach().abs().sum()), 0.)

    def test_temporal_zero_head_preserves_future(self):
        from djepa.temporal_transport import ReacherFutureResidualHead
        model = ReacherFutureResidualHead()
        future = torch.randn(1, 3, 5, 192)
        ranks = torch.tensor([[0., .5, 1.]])
        out = model(tdjepa_future=future, lewm_future=future+1, lewm_rank=ranks,
                    tdjepa_rank=ranks, target_rank=ranks, gate_active=torch.tensor([True]))
        torch.testing.assert_close(out.future, future, atol=0, rtol=0)

    def test_exact_lifting_keeps_prefix_and_recovers_order(self):
        from djepa.inference import exact_lift
        future = torch.zeros(1, 3, 5, 192)
        goal = torch.zeros(1, 192)
        out = exact_lift(future, goal, torch.tensor([[.5, .1, .1]]), torch.tensor([[7, 9, 2]]))
        torch.testing.assert_close(out[:, :, :4], future[:, :, :4], atol=0, rtol=0)
        native = (out[:, :, -1]-goal[:, None]).square().mean(-1)
        self.assertEqual(native.argmin(1).item(), 2)
        torch.testing.assert_close(native, torch.tensor([[9., 4., 1.]])/4096)

    def test_native_cost_uses_terminal_only(self):
        from djepa.native_scoring import native_terminal_cost
        future = torch.ones(1, 2, 5, 192)
        future[:, :, :4] = 100
        torch.testing.assert_close(native_terminal_cost(future, torch.zeros(1, 192)), torch.ones(1, 2))

    def test_unobserved_labels_are_not_counted_as_failures(self):
        from djepa.evaluate import summarize_selections
        result = summarize_selections(np.array([0, 1]), np.array([[True, False], [False, False]]),
                                      np.array([[True, False], [True, False]]))
        self.assertEqual(result['observed_count'], 1)
        self.assertEqual(result['success_count'], 1)
        self.assertEqual(result['unobserved_count'], 1)

    def test_training_rejects_overlapping_ids(self):
        from djepa.train import require_disjoint
        with self.assertRaises(ValueError):
            require_disjoint(np.array(['a','b']), np.array(['b','c']))

    def test_profile_roundtrip_predicts_without_labels(self):
        from djepa.ordinal import DecisionAligner
        from djepa.inference import predict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps({'architecture':'set_aligner','input_dim':3,
                'gate_mode':'refined_gap','gate_threshold':.01}))
            torch.save(DecisionAligner(3).state_dict(), root/'model.pt')
            result=predict(root,{'features':np.zeros((1,3,3)), 'base_scores':np.array([[.5,.2,.3]]),
                                 'candidate_ids':np.array([[10,20,30]])})
            self.assertEqual(result['selected_ids'].tolist(), [20])

    def test_composition_retains_default_unless_proposal_clears_gate(self):
        from djepa.inference import compose_decisions
        base=np.array([[.4,.2,.3],[.4,.2,.3]])
        refined=np.array([[.1,.3,.4],[.1,.3,.4]])
        ids=np.array([[1,2,3],[1,2,3]])
        proposals=np.array([[.095,.4,.5],[.01,.4,.5]])
        result=compose_decisions(base,refined,ids,np.array([0,0]),proposals,.0065339543)
        np.testing.assert_array_equal(result,[1,1])
        proposals=np.array([[.3,.2,.095],[.3,.2,.01]])
        result=compose_decisions(base,refined,ids,np.array([0,0]),proposals,.0065339543)
        np.testing.assert_array_equal(result,[1,3])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
