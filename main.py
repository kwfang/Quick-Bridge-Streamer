#!/usr/bin/env python3
"""
OpenX-Explorer (oxview) - BridgeData V2 Dataset Viewer

Usage:
  python main.py data --limit 100
  python main.py data --dataset 1 --limit 100  # Select the 1st dataset
  python main.py data --no-interactive         # Non-interactive mode (auto-select first)

Automatically opens browser with real-time streaming to Rerun Web Viewer.
No rerun-cli installation needed, no manual file dragging required.
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        print(f"[INFO] TensorFlow version: {tf.__version__}")
    except ImportError:
        print("[ERROR] TensorFlow not found. Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        import rerun as rr
        print(f"[INFO] Rerun SDK version: {rr.__version__}")
    except ImportError:
        print("[ERROR] Rerun SDK not found. Please run: pip install -r requirements.txt")
        sys.exit(1)


def print_banner():
    """Print startup banner"""
    print("""
============================================================
    OpenX-Explorer - BridgeData V2 Edition
    
    Visualize robot datasets without PyTorch.
    Just: pip install -r requirements.txt && python main.py data
============================================================
""")


def main():
    """Main function"""
    import click
    from oxview.loader import BridgeDataLoader, DatasetScanner
    from oxview.viewer import RerunApp
    
    @click.command()
    @click.argument('data_root', type=click.Path(exists=True, path_type=Path))
    @click.option('--limit', default=-1, help='Limit number of frames (-1 = all)')
    @click.option('--split', default='train', help='Data split (train/val)')
    @click.option('--dataset', '-d', default=None, type=int, help='Dataset index to load (1-based, auto-scan and select if not specified)')
    @click.option('--no-interactive', is_flag=True, help='Non-interactive mode (auto-select first dataset)')
    def cli(data_root: Path, limit: int, split: str, dataset: int, no_interactive: bool):
        """
        BridgeData V2 Dataset Viewer
        
        DATA_ROOT: Root folder containing dataset(s). Can be:
          - A folder with multiple dataset subfolders
          - A folder containing .tfrecord files directly
        """
        print_banner()
        
        data_root = data_root.resolve()
        print(f"[INFO] Data root: {data_root}")
        
        # Scan available datasets
        scanner = DatasetScanner(data_root)
        datasets = scanner.scan()
        
        if not datasets:
            print(f"[ERROR] No datasets found in {data_root}")
            print("[INFO] Expected folder structure:")
            print("  Option 1: data_root/dataset_a/*.tfrecord-*")
            print("  Option 2: data_root/*.tfrecord-*")
            sys.exit(1)
        
        print(f"[INFO] Found {len(datasets)} dataset(s)")
        
        # Select dataset
        if dataset is not None:
            # User specified via --dataset argument
            if dataset < 1 or dataset > len(datasets):
                print(f"[ERROR] Invalid dataset index: {dataset} (valid: 1-{len(datasets)})")
                sys.exit(1)
            selected = datasets[dataset - 1]
            print(f"[INFO] Selected dataset [{dataset}]: {selected['name']}")
        else:
            # Interactive or non-interactive selection
            selected = scanner.select_dataset(datasets, interactive=not no_interactive)
            if selected is None:
                sys.exit(1)
            print(f"[INFO] Selected dataset: {selected['name']}")
        
        dataset_path = selected['path']
        
        # Display available dataset list to Rerun (via text log)
        # This will be recorded as static data in the recording
        try:
            import rerun as rr
            ds_list_md = "# Available Datasets\\n\\n"
            for i, ds in enumerate(datasets, 1):
                marker = "✓ " if ds['name'] == selected['name'] else "  "
                ds_list_md += f"{marker}[{i}] **{ds['name']}** ({ds['num_files']} files)\\n"
            ds_list_md += f"\\n_Currently viewing: **{selected['name']}**_"
            
            # Initialize rerun first to log static information
            rr.init("OpenX-Explorer")
            rr.log("info/datasets", rr.TextDocument(ds_list_md, media_type="text/markdown"), static=True)
        except Exception as e:
            # If this fails, it doesn't affect the main flow
            print(f"[WARNING] Could not log dataset list: {e}")
        
        # Initialize loader
        try:
            loader = BridgeDataLoader(dataset_path, split=split)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        
        stats = loader.get_statistics()
        print(f"[INFO] Found {stats['num_tfrecord_files']} TFRecord files")
        print(f"[INFO] Backend: TensorFlow-CPU")
        print()
        
        # Launch viewer (full auto: web streaming + auto browser)
        app = RerunApp(loader, limit=limit)
        app.run()
    
    cli()


if __name__ == "__main__":
    check_dependencies()
    main()
