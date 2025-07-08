
from train import APFTrainer
from data import fps_sampling_with_knn_optimized

model_config = {
    'model_name': 'AdaptPointFormerWithSampling',
    'num_classes': 15,
    'num_points': 2048,
    'vit_name': 'vit_b_16',
    'adapter_dim': 64,
    'pretrained': True,
    'dropout_rate': 0.1,
    'fps_sampling_func': fps_sampling_with_knn_optimized,
    'n_samples': 512,
    'k_neighbors': 16,
}

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
    'learning_rate': 5e-4,
    'weight_decay': 5e-2,
    'save_interval': 5,
    'epochs': 100,
}

trainer = APFTrainer(
    model_config=model_config,
    dataset_config=dataset_config,
    train_config=train_config,
    device='cuda'
)

trainer.train()
