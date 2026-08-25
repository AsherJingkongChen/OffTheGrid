import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from src.utils.load_utils import load_and_preprocess_images

def main():
    input_dir = Path("minimal_dataset")
    output_dir = Path("minimal_dataset2")
    
    if not input_dir.exists():
        print(f"Error: {input_dir} not found.")
        return
        
    # Iterate through all files in input_dir
    for file_path in input_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(input_dir)
            
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Image file: resize and convert to png
                out_path = output_dir / rel_path
                out_path = out_path.with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Processing image {file_path} -> {out_path}")
                try:
                    # load_and_preprocess_images returns a batched tensor: (1, C, H, W)
                    tensor_images = load_and_preprocess_images(
                        [str(file_path)], 
                        mode='vggt_crop', 
                        target_size=518
                    )
                    
                    # Extract the single image from the batch
                    img_tensor = tensor_images[0].permute(1, 2, 0).cpu().numpy()
                    img_np = (np.clip(img_tensor, 0, 1) * 255).astype(np.uint8)
                    img = Image.fromarray(img_np)
                    img.save(str(out_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                # Other file: just copy it to preserve dataset structure (e.g. JSON, txt, npy)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Copying file {file_path} -> {out_path}")
                shutil.copy2(file_path, out_path)

if __name__ == "__main__":
    main()
