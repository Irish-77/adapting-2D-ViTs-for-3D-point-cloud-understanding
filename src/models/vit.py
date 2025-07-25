import timm
import torch.nn as nn
import torchvision.models as models
from typing import Optional

def get_vit(
    vit_name: str, 
    pretrained: bool = True
) -> tuple[models.VisionTransformer, int]:
    """
    Load a Vision Transformer model by name
    
    Args:
        vit_name: Name of the ViT model to load
        pretrained: Whether to load pretrained weights
    
    Returns:
        model: The loaded ViT model
        embed_dim: The embedding dimension of the model
    """
    
    if vit_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
        embed_dim = 768
    elif vit_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=pretrained)
        embed_dim = 768
    elif vit_name == 'vit_l_16':
        model = models.vit_l_16(pretrained=pretrained)
        embed_dim = 1024
    elif vit_name == 'vit_l_32':
        model = models.vit_l_32(pretrained=pretrained)
        embed_dim = 1024
    else:
        raise ValueError(f"Unsupported ViT model: {vit_name}")
    
    return model, embed_dim


def get_timm_vit(
    name: str,
    pretrained: bool = True,
    delete: Optional[list[str]] = None,
) -> nn.Module:
    """
    Load a Vision Transformer model from timm library by name
    
    Args:
        name: Name of the ViT model to load
        pretrained: Whether to load pretrained weights
        delete: Optional parameter containing specific layers or attributes to delete from the state_dict
    
    Returns:
        model: The loaded ViT model; if delete is not None, state_dict is returned
    """
    
    model = timm.create_model(name, pretrained=pretrained)
    if delete is None:
        return model

    state_dict = model.state_dict()
    for d in delete:
        if d in state_dict:
            del state_dict[d]

    return state_dict