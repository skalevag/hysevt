import logging

logging.basicConfig(
    filename=f"{__name__}.log",
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.ERROR,
)