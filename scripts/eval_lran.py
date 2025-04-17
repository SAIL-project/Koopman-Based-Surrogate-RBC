import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from koopman_rbc.data import RBCDatamodule
from koopman_rbc.models import LRANModule
from koopman_rbc.utils.callbacks import (
    MetricsCallback,
    SequenceExamplesCallback,
    SequenceMetricsCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="lran.yaml")
def main(cfg: DictConfig):
    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-LRAN",
        save_dir=cfg.paths.output_dir,
        log_model=False,
        tags=["eval"],
    )

    # data
    dm = RBCDatamodule(**cfg.data)

    # model
    ckpt = f"{cfg.paths.root_dir}/models/lran/ra{cfg.data.ra}.ckpt"
    model = LRANModule.load_from_checkpoint(ckpt)
    model.eval()

    # callbacks
    callbacks = [
        MetricsCallback(name="metric", key_groundtruth="x", key_prediction="x_hat"),
        SequenceMetricsCallback(
            name="metric", key_groundtruth="x", key_prediction="x_hat"
        ),
        SequenceExamplesCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=cfg.paths.output_dir,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=cfg.epochs,
        callbacks=callbacks,
    )

    # rollout on validation set
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
