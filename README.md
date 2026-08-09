        # MMA-Net Code and Data

        Training code, model configuration, and YOLO-format datasets for **MMA-Net**
        steel-surface defect detection (NEU-DET, GC10-DET, Crack-Seg).

        ## Repository structure

        ```
        .
        ├── train.py              # Main training entry (EnhancedAMSDALoss)
        ├── train_with_amsdal.py  # Alternate training script
        ├── new_moudle.yaml       # MMA-Net architecture (LMDAM + LEFPN + LSCAM)
        ├── biaozhun2.yaml        # YOLO11n baseline
        ├── data.yaml             # Default NEU-DET config for train.py
        ├── configs/              # Dataset YAML configs
        ├── ultralytics/          # Modified Ultralytics fork (custom modules + loss)
        └── datasets/
        - `datasets/NEU-DET/`
- `datasets/GC10-DET/`
- `datasets/Crack-Seg/`
        ```

        ## Quick start

        ```bash
        git clone https://github.com/gg123654/code--and---data.git
        cd code--and---data
        pip install -e .

        # Train MMA-Net on NEU-DET (default)
        python train.py

        # Train on GC10-DET
        python -c "from ultralytics import YOLO; YOLO('new_moudle.yaml').train(data='configs/gc10_det.yaml', epochs=500, imgsz=640, batch=16)"
        ```

        ## Requirements

        - Python >= 3.8
        - PyTorch + CUDA (recommended)
        - See `pyproject.toml` for dependencies

        ## Citation

        If you use this code or data, please cite the MMA-Net paper and the original
        NEU-DET / GC10-DET / Crack-Seg benchmark papers.
