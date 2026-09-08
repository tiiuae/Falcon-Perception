import unittest

import torch

from falcon_perception.paged_inference import Sequence


class SequenceOutputTest(unittest.TestCase):
    def test_output_contract_without_scalar_extraction(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for count in (0, 1, 17):
                with self.subTest(device=device, count=count):
                    seq = Sequence(text="", image=None)
                    ids = torch.arange(count, device=device)
                    logits = torch.linspace(
                        -4, 4, count, device=device, dtype=torch.float16
                    )
                    probs = torch.linspace(
                        0, 1, count, device=device, dtype=torch.bfloat16
                    )
                    logits.requires_grad_()
                    probs.requires_grad_()
                    seq._output_ids = list(ids.unbind())
                    seq._output_logits = list(logits.unbind())
                    seq._output_probs = list(probs.unbind())
                    seq.input_ids = torch.tensor([99, 100])
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profile:
                        outputs = (seq.output_ids, seq.output_logits, seq.output_probs)
                    scalar_calls = sum(
                        e.count
                        for e in profile.key_averages()
                        if e.key == "aten::_local_scalar_dense"
                    )
                    self.assertEqual(scalar_calls, 0)
                    for actual, expected in zip(
                        outputs, (ids.cpu(), logits.float().cpu(), probs.float().cpu())
                    ):
                        self.assertEqual(actual.device.type, "cpu")
                        self.assertFalse(actual.requires_grad)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(
                        torch.equal(
                            seq.total_token_ids, torch.cat((seq.input_ids, ids.cpu()))
                        )
                    )
                    if count:
                        outputs[0][0] = -1
                        self.assertEqual(seq.output_ids[0].item(), 0)


if __name__ == "__main__":
    unittest.main()
