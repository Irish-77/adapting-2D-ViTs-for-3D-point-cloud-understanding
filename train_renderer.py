from train import RendererTrainer

dataset_config = {
    'root_dir': '/home/basti/Development/University/3DVision/adapting-2D-ViTs-for-3D-point-cloud-understanding/.data/h5_files',
    'split': 'training',
    'variant': 'main_split',
    'augmentation': 'base',
    'num_points': 2048,
    'normalize': True,
    'sampling_method': 'all',
    'use_custom_augmentation': True,
}

train_config = {
    'batch_size': 16,
    'min_lr': 1e-6,
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'save_interval': 5,
    'epochs': 100,
    'use_lr_scheduler': False,
    'clip_grad_norm': 0.0,
}

model_config = {
    'num_classes': 15,
    'vit_name': 'vit_b_16',
    'adapter_dim': 64,
    'num_views': 6,
    'img_size': 224,
    'pretrained': True, 
    'dropout_rate': 0.1,
    'diff_renderer': True,
    'view_transform_hidden': 256,
}

trainer = RendererTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda',
    output_dir='./output/renderer_training_diff' 
)

trainer.train()