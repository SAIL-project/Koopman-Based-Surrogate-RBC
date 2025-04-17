import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from koopman_rbc.data import RBCDatamodule
from koopman_rbc.models import AutoencoderLitModule
from koopman_rbc.utils.callbacks import ExamplesCallback, MetricsCallback


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="autoencoder.yaml"
)
def main(cfg: DictConfig):
    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-AE",
        save_dir=cfg.paths.output_dir,
        log_model=False,
        tags=["eval"],
    )
    logger.experiment.config.update(dict(cfg.data))

    # data
    dm = RBCDatamodule(**cfg.data)

    # model
    ckpt = f"{cfg.paths.root_dir}/models/autoencoder/ra{cfg.data.ra}.ckpt"
    model = AutoencoderLitModule.load_from_checkpoint(ckpt)
    model.eval()

    # callbacks
    callbacks = [
        MetricsCallback(
            name="metric",
            key_groundtruth="x",
            key_prediction="x_hat",
        ),
        ExamplesCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        precision="16-mixed",
        default_root_dir=cfg.paths.output_dir,
        deterministic=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        callbacks=callbacks,
    )

    # training
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
