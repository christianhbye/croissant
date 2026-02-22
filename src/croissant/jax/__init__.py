import logging

logger = logging.getLogger(__name__)
logger.warning(
    "The croissant.jax interface is deprecated and will be removed in a "
    "future release. Please use the croissant interface directly instead.",
)

from .. import *
