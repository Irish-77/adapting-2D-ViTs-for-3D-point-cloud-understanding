from train import Pix4PointTrainer


model_config = {
    'model_name': 'Pix4Point',
    'num_classes': 15,
    'pretrained_model': 'vit_small_patch16_384.augreg_in21k_ft_in1k', # 'vit_base_patch16_384.augreg_in21k_ft_in1k'
    'k_neighbors': 16,
    'embed_dim': 384 # 678
}

dataset_config = {
    'root_dir': './.data/h5_files',
    'split': 'training',
    'variant': 'main_split',
    'augmentation': 'base',
    'num_points': 2048,
    'normalize': True,
    'sampling_method': 'all',
    'use_custom_augmentation': True,
}

train_config = {
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'save_interval': 5,
    'epochs': 100,
}

trainer = Pix4PointTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda'
)

trainer.train()
