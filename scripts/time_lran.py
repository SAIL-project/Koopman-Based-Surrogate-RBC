import os

import hydra
import lightning as L
import psutil
from omegaconf import DictConfig

from koopman_rbc.data import RBCDatamodule
from koopman_rbc.models import LRANModule
from koopman_rbc.utils.callbacks import MemoryCallback, TimerCallback


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2


@hydra.main(version_base="1.3", config_path="../configs", config_name="lran.yaml")
def main(cfg: DictConfig):
    # memory
    mem_start = process_memory()
    print(f"Memory Start: {mem_start:.2f} MB")

    # data
    dm = RBCDatamodule(**cfg.data)

    # model
    ckpt = f"{cfg.paths.root_dir}/models/lran/ra{cfg.data.ra}.ckpt"
    model = LRANModule.load_from_checkpoint(ckpt)
    model.eval()

    # callbacks
    callbacks = [
        TimerCallback(),
        MemoryCallback(start_memory=mem_start),
    ]

    # trainer
    trainer = L.Trainer(
        accelerator="cpu",
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
