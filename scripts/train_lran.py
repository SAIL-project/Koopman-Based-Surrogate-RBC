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
from koopman_rbc.models import LRANModule
from koopman_rbc.utils.callbacks import (
    MetricsCallback,
    SequenceExamplesCallback,
    SequenceMetricsCallback,
)
from koopman_rbc.utils.transforms import NormalizeInverse


@hydra.main(version_base="1.3", config_path="../configs", config_name="lran.yaml")
def main(cfg: DictConfig):
    # data
    dm = RBCDatamodule(**cfg.data)

    # model
    inv_transform = NormalizeInverse(mean=cfg.data.means, std=cfg.data.stds)
    model = LRANModule(**cfg.model, inv_transform=inv_transform)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-LRAN",
        save_dir=cfg.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        MetricsCallback(name="metric", key_groundtruth="x", key_prediction="x_hat"),
        SequenceMetricsCallback(
            name="metric", key_groundtruth="x", key_prediction="x_hat"
        ),
        SequenceExamplesCallback(train_freq=3),
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
        default_root_dir=cfg.paths.output_dir,
        deterministic=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=cfg.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
