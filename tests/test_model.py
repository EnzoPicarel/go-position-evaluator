import unittest
import numpy as np
import torch
from go_model import stones_to_board, board_symmetries, split_samples, position_key, GoBoardCNN, predict


class ModelTests(unittest.TestCase):
    def test_coordinate_encoding(self):
        b = stones_to_board(['A1','H8'], ['B3'])
        self.assertEqual(b.shape, (2,8,8))
        self.assertEqual(b[0,0,0], 1)
        self.assertEqual(b[0,7,7], 1)
        self.assertEqual(b[1,2,1], 1)
        self.assertEqual(b.sum(), 3)

    def test_symmetries_preserve_stones_and_label(self):
        b = stones_to_board(['A1','B3'], ['H2'])
        variants = board_symmetries(b, .7)
        self.assertEqual(len(variants), 8)
        for x, y in variants:
            np.testing.assert_array_equal(x.sum(axis=(1,2)), [2,1])
            self.assertEqual(y, .7)

    def test_split_keeps_equivalent_positions_together(self):
        samples = [dict(black_stones=['A1'], white_stones=[]),
                   dict(black_stones=['H8'], white_stones=[]),
                   dict(black_stones=['B2'], white_stones=[]),
                   dict(black_stones=['C3'], white_stones=[])]
        train, valid = split_samples(samples)
        self.assertFalse({position_key(s) for s in train} & {position_key(s) for s in valid})
        self.assertEqual(len(train)+len(valid), len(samples))
        self.assertEqual((train,valid), split_samples(samples))

    def test_inference_and_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path
        torch.set_num_threads(2)
        model = GoBoardCNN()
        samples = [dict(black_stones=['A1'], white_stones=['B2'])]
        expected = predict(model,samples)
        self.assertTrue(np.all((expected>=0)&(expected<=1)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            torch.save(model.state_dict(),path)
            copy=GoBoardCNN()
            copy.load_state_dict(torch.load(path,weights_only=True))
            np.testing.assert_allclose(expected,predict(copy,samples))


if __name__ == '__main__':
    unittest.main()
