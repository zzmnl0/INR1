"""
损失函数模块
"""
from .physics_losses import (
    heteroscedastic_loss,
    iri_gradient_direction_loss,
    tec_gradient_alignment_loss,
    smoothness_loss_tv,
    get_obs_weight,
    get_background_trust
)

__all__ = [
    'heteroscedastic_loss',
    'iri_gradient_direction_loss',
    'tec_gradient_alignment_loss',
    'smoothness_loss_tv',
    'get_obs_weight',
    'get_background_trust'
]
