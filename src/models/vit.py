import torchvision.models as models

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