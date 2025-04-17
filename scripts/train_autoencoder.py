import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig


from koopman_rbc.data import RBCDatamodule
from koopman_rbc.models import AutoencoderLitModule, VariationalAutoencoderLitModule
from koopman_rbc.utils.callbacks import MetricsCallback, ExamplesCallback
from koopman_rbc.utils.transforms import NormalizeInverse


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="autoencoder.yaml"
)
def main(cfg: DictConfig):
    # data
    dm = RBCDatamodule(**cfg.data)

    # model
    inv_transform = NormalizeInverse(mean=cfg.data.means, std=cfg.data.stds)
    if cfg.modeltype == "ae":
        model = AutoencoderLitModule(**cfg.model, inv_transform=inv_transform)
    elif cfg.modeltype == "vae":
        model = VariationalAutoencoderLitModule(
            **cfg.model, inv_transform=inv_transform
        )
    else:
        raise ValueError(f"Model type {cfg.modeltype} not supported")

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-AE",
        save_dir=cfg.paths.output_dir,
        log_model=False,
    )
    logger.experiment.config.update(dict(cfg.data))
    print(f"Save logs and checkpoints to {cfg.paths.output_dir}")

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        MetricsCallback(
            name="metric",
            key_groundtruth="x",
            key_prediction="x_hat",
        ),
        ExamplesCallback(train_freq=5, test_freq=5),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=10,
        ),
        ModelCheckpoint(
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
        ),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        precision="16-mixed",
        default_root_dir=cfg.paths.output_dir,
        enable_model_summary=False,
        deterministic=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=cfg.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
