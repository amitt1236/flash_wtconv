"""CPU regression checks for the paired Tiny ImageNet training recipe.

Run with: python tests/cuda_metal_tests/test_train_recipe.py
"""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import test_train_convergence as training


class TrainingRecipeTests(unittest.TestCase):
    def test_warmup_and_cosine(self):
        rates = [training.learning_rate_for_epoch(e, 50, 3e-4) for e in range(50)]
        for actual, expected in zip(rates[:4], [1e-4, 2e-4, 3e-4, 3e-4]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(rates[-1], 1e-6)
        self.assertTrue(all(a >= b for a, b in zip(rates[3:], rates[4:])))
        self.assertEqual(training.learning_rate_for_epoch(0, 1, 3e-4), 3e-4)
        self.assertAlmostEqual(training.learning_rate_for_epoch(4, 5, 3e-4, 0), 1e-6)
        for epochs in (2, 3):
            rates = [training.learning_rate_for_epoch(e, epochs, 3e-4) for e in range(epochs)]
            self.assertTrue(all(0 < rate <= 3e-4 for rate in rates))

    def test_invalid_schedule(self):
        for kwargs in ({'epochs': 0}, {'warmup_epochs': -1}, {'min_lr': 1.0}):
            config = dict(epoch=0, epochs=50, lr=3e-4)
            config.update(kwargs)
            with self.assertRaises(ValueError):
                training.learning_rate_for_epoch(**config)

    def test_optimizer_groups_match_between_implementations(self):
        counts = []
        for cls in (training.WTConv2d, training.WTConv2dNaive):
            model = training.build_model(cls)
            optimizer = training.build_optimizer(model)
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            decay, no_decay = optimizer.param_groups
            decayed = {id(p) for p in decay['params']}
            excluded = {id(p) for p in no_decay['params']}
            self.assertFalse(decayed & excluded)
            self.assertEqual(decayed | excluded, {id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual(decay['weight_decay'], 0.05)
            self.assertEqual(no_decay['weight_decay'], 0)
            for stage in model.stages:
                for block in stage.blocks:
                    self.assertIn(id(block.gamma), excluded)
                    self.assertIn(id(block.norm.weight), excluded)
                    self.assertIn(id(block.mlp.fc1.weight), decayed)
                    self.assertIn(id(block.mlp.fc1.bias), excluded)
                    weight, bias, scale, wt_weights, wt_scales = training._wtconv_tensors(block.conv_dw)
                    for p in [weight, *wt_weights]:
                        self.assertIn(id(p), decayed)
                    for p in [bias, scale, *wt_scales]:
                        self.assertIn(id(p), excluded)
            counts.append([sum(p.numel() for p in g['params']) for g in optimizer.param_groups])
        self.assertEqual(*counts)

    def test_augmented_batches_repeat_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'tiny-imagenet-200'
            folder = root / 'train' / 'n001' / 'images'
            folder.mkdir(parents=True)
            # Spatially varying images make differences in crops/flips observable.
            pixels = torch.arange(64 * 64 * 3).remainder(251).byte().reshape(64, 64, 3).numpy()
            image = Image.fromarray(pixels)
            for i in range(4):
                image.save(folder / f'{i}.JPEG')
            val_folder = root / 'val' / 'images'
            val_folder.mkdir(parents=True)
            image.save(val_folder / 'val.JPEG')
            (root / 'val' / 'val_annotations.txt').write_text('val.JPEG\tn001\t0\t0\t64\t64\n')
            for workers in (0, 2):
                def collect(seed):
                    torch.manual_seed(seed)
                    train, val = training.get_tiny_imagenet_loaders(
                        batch_size=2, num_workers=workers, data_root=tmp, seed=seed,
                    )
                    batches = [torch.cat([x for x, _ in train]) for _ in range(2)]
                    return batches, val.dataset[0][0]
                first, validation = collect(42)
                second, _ = collect(42)
                other, other_validation = collect(43)
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertFalse(torch.equal(first[0], first[1]))
                self.assertFalse(torch.equal(first[0], other[0]))
                torch.testing.assert_close(validation, other_validation, rtol=0, atol=0)

    def test_training_replays_rng_and_logs_schedule(self):
        dataset = TensorDataset(torch.randn(8, 4), torch.arange(8).remainder(2))
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        results = []
        states = []
        for _ in range(2):
            model.load_state_dict(initial)
            train = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator())
            val = DataLoader(dataset, batch_size=4)
            results.append(training.train_model(model, train, val, 5, 3e-4, torch.device('cpu'), seed=42))
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        for key in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'):
            self.assertEqual(results[0][key], results[1][key])
        self.assertAlmostEqual(results[0]['lr'][-1], 1e-6)
        self.assertFalse(torch.equal(initial['0.weight'], states[0]['0.weight']))
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
