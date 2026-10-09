from __future__ import annotations

import sys
#sys.path.append(".")
#sys.path.append("/models/voice_conversion/so-vits-svc/src/so_vits_svc_fork")

import os
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    StreamHandler,
    basicConfig,
    captureWarnings,
    getLogger,
)
from pathlib import Path
from typing import Literal

import click
import pyinputplus as pyip
import torch
from rich.logging import RichHandler

print("Env vars:")
print(os.environ)


def print_gpu_info():
    print('========================================')
    print('Python interpreter', sys.executable)
    print('PyTorch version', torch.__version__)
    print('CUDA Available?', torch.cuda.is_available())
    print('CUDA Device count', torch.cuda.device_count())
    print('CUDA architectures library was compiled for', torch.cuda.get_arch_list())

    #try:
    #    from tensorflow.python.client import device_lib
    #    print('local devices', str(device_lib.list_local_devices()).replace("\n", "\n  "))
    #except ImportError:
    #    print('no tensorflow - cannot list devices')
    #    pass
    print('========================================', flush=True)

print_gpu_info()

def init_logger() -> None:
    IN_COLAB = os.getenv("COLAB_RELEASE_TAG")
    IS_TEST = "test" in Path(__file__).parent.stem

    basicConfig(
        level=DEBUG if IS_TEST else INFO,
        format="%(asctime)s %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler() if not IN_COLAB else StreamHandler(),
            FileHandler(f"{__name__.split('.')[0]}.log"),
        ],
    )
    captureWarnings(True)
    if IS_TEST:
        LOG.debug("Test mode is on.")


init_logger()

LOG = getLogger(__name__)


class RichHelpFormatter(click.HelpFormatter):
    def __init__(
        self,
        indent_increment: int = 2,
        width: int | None = None,
        max_width: int | None = None,
    ) -> None:
        width = 100
        super().__init__(indent_increment, width, max_width)


def patch_wrap_text():
    orig_wrap_text = click.formatting.wrap_text

    def wrap_text(
        text,
        width=78,
        initial_indent="",
        subsequent_indent="",
        preserve_paragraphs=False,
    ):
        return orig_wrap_text(
            text.replace("\n", "\n\n"),
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            preserve_paragraphs=True,
        ).replace("\n\n", "\n")

    click.formatting.wrap_text = wrap_text


patch_wrap_text()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)
click.Context.formatter_class = RichHelpFormatter


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """so-vits-svc allows any folder structure for training data.
    However, the following folder structure is recommended.\n
        When training: dataset_raw/{speaker_name}/**/{wav_name}.{any_format}\n
        When inference: configs/44k/config.json, logs/44k/G_XXXX.pth\n
    If the folder structure is followed, you DO NOT NEED TO SPECIFY model path, config path, etc.
    (The latest model will be automatically loaded.)\n
    To train a model, run pre-resample, pre-config, pre-hubert, train.\n
    To infer a model, run infer.
    """
    init_logger()



@cli.command()
@click.argument(
    "input-path",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    help="path to output dir",
)
@click.option("-s", "--speaker", type=str, default=None, help="speaker name")
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    default=Path("./logs/44k/"),
    help="path to model",
)
@click.option(
    "-c",
    "--config-path",
    type=click.Path(exists=True),
    default=Path("./configs/44k/config.json"),
    help="path to config",
)
@click.option(
    "-k",
    "--cluster-model-path",
    type=click.Path(exists=True),
    default=None,
    help="path to cluster model",
)
@click.option("-t", "--transpose", type=int, default=0, help="transpose")
@click.option(
    "-db", "--db-thresh", type=int, default=-20, help="threshold (DB) (RELATIVE)"
)
@click.option(
    "-a", "--auto-predict-f0", type=bool, default=True, help="auto predict f0"
)
@click.option(
    "-r", "--cluster-infer-ratio", type=float, default=0, help="cluster infer ratio"
)
@click.option("-n", "--noise-scale", type=float, default=0.4, help="noise scale")
@click.option("-p", "--pad-seconds", type=float, default=0.5, help="pad seconds")
@click.option(
    "-d",
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device",
)
@click.option("-ch", "--chunk-seconds", type=float, default=0.5, help="chunk seconds")
@click.option(
    "-ab", "--absolute-thresh", type=bool, default=False, help="absolute thresh"
)
def infer(
    input_path: Path,
    output_path: Path,
    speaker: str,
    model_path: Path,
    config_path: Path,
    cluster_model_path: Path | None = None,
    transpose: int = 0,
    db_thresh: int = -40,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    pad_seconds: float = 0.5,
    chunk_seconds: float = 0.5,
    absolute_thresh: bool = False,
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Inference"""
    from .inference_main import infer
    #from inference_main import infer

    if not auto_predict_f0:
        LOG.warning(
            f"auto_predict_f0 = False, transpose = {transpose}. If you want to change the pitch, please set transpose."
            "Generally transpose = 0 does not work because your voice pitch and target voice pitch are different."
        )

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.out{input_path.suffix}"
    output_path = Path(output_path)
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = list(sorted(model_path.glob("*.pth")))[-1]
        LOG.info(f"Since model_path is a directory, use {model_path}")
    config_path = Path(config_path)
    if cluster_model_path is not None:
        cluster_model_path = Path(cluster_model_path)
    
    LOG.info("Calling infer()...")

    infer(
        input_path=input_path,
        output_path=output_path,
        speaker=speaker,
        model_path=model_path,
        config_path=config_path,
        cluster_model_path=cluster_model_path,
        transpose=transpose,
        db_thresh=db_thresh,
        auto_predict_f0=auto_predict_f0,
        cluster_infer_ratio=cluster_infer_ratio,
        noise_scale=noise_scale,
        pad_seconds=pad_seconds,
        chunk_seconds=chunk_seconds,
        absolute_thresh=absolute_thresh,
        device=device,
    )



if __name__ == '__main__':
    infer()
