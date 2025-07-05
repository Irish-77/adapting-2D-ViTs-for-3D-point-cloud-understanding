import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data import ScanObjectNN
from models.renderer import PointCloudRenderer


def main():
    train_dataset = ScanObjectNN(
        root_dir='/home/basti/Development/University/3DVision/adapting-2D-ViTs-for-3D-point-cloud-understanding/.data/h5_files',
        split='training',
        variant='main_split',
        augmentation='base',
        num_points=2048,
        normalize=True,
        sampling_method='all',
        use_custom_augmentation=True,
    )

    pc_render = PointCloudRenderer(
        img_size=224,
        num_views=6,
    )
    pc_render.to("cuda")

    point_cloud, label = train_dataset[0]
    point_cloud = torch.tensor(point_cloud).unsqueeze(0)

    point_cloud.to("cuda")

    import time
    start_time = time.time()
    with torch.no_grad():
        rendered_views = pc_render(point_cloud)  # [1, num_views, 3, H, W]
    print(f"Rendering time: {time.time() - start_time:.2f} seconds")    

    rendered_views = rendered_views.cpu()  # Move back to CPU for visualization

    # Plot all views
    fig, axes = plt.subplots(1, pc_render.num_views, figsize=(15, 3))

    for i in range(pc_render.num_views):
        # Get the rendered image for this view
        img_tensor = rendered_views[0, i]  # [3, H, W]
        
        # Convert to numpy and adjust format for PIL

        img_np = img_tensor.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, 3]
        
        # Normalize to [0, 255] range
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255
        img_np = img_np.astype(np.uint8)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_np)
        
        # Display in matplotlib
        axes[i].imshow(img_pil)
        axes[i].set_title(f"View {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('rendered_views.png')
    plt.show()


if __name__ == "__main__":
    main()