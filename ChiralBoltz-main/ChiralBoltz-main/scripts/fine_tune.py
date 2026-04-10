#!/usr/bin/env python
"""
Main entry point for ChiralBoltz fine-tuning.

Usage:
    python scripts/fine_tune.py --config configs/chiralboltz_finetune.yaml

Requires:
  - Boltz-2 weights downloaded via `boltz predict` (cached in ~/.boltz)
  - Processed PDB NPZ structures and MSAs (see docs/training.md in jwohlwend/boltz)
  - Optionally: mirrored NPZ structures from prepare_mirror_dataset.py
"""
import argparse
import yaml
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from chiralboltz.training.config import ChiralBoltzConfig
from chiralboltz.training.trainer import ChiralBoltzModule


def load_config(path: str) -> ChiralBoltzConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    return ChiralBoltzConfig(**d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/chiralboltz_finetune.yaml")
    parser.add_argument("--wandb",  action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load Boltz-2
    try:
        from boltz.model.models.boltz2 import Boltz2
        boltz2_model = Boltz2.load_from_checkpoint(cfg.boltz2_checkpoint) if cfg.boltz2_checkpoint else Boltz2()
    except ImportError:
        raise ImportError("boltz is not installed. Run: pip install boltz")

    module = ChiralBoltzModule(boltz2_model=boltz2_model, config=cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=Path(cfg.output_dir) / "checkpoints",
            filename="chiralboltz-{step:06d}-{val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    loggers = []
    if args.wandb:
        loggers.append(WandbLogger(project="chiralboltz", name="mirror-aug-finetune"))

    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        devices=cfg.devices,
        precision=cfg.precision,
        callbacks=callbacks,
        logger=loggers or True,
        log_every_n_steps=50,
        val_check_interval=1000,
        default_root_dir=cfg.output_dir,
    )

    # TODO: wire up DataModule (MirrorAugmentedDataset + Boltz-2 DataModule)
    # trainer.fit(module, datamodule=...)
    print(f"ChiralBoltz module ready. Config: {cfg}")
    print("To train: wire up a Boltz-2 DataModule with MirrorAugmentedDataset wrapper.")


if __name__ == "__main__":
    main()
