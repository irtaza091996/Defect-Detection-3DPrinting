"""Extract individual B-scan frames from 3D OCT TIFF volumes and save as PNG images.

Optionally launch an interactive Jupyter slider widget to browse frames.

Usage (script):
    python Preprocessing/tiff_processing.py --tiff-dir /path/to/tiff --out-dir /path/to/output

Usage (Jupyter):
    viewer = OCTViewer("A-09_1", "path/to/Timing_A-09_1_layer.tif", "output/")
    display(viewer.show_slider())
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile as tiff


DATASETS = [
    ("A-09_1", "Timing_A-09_1_layer.tif",      "Saved_Frames_A-09_1"),
    ("A-09_2", "Timing_A-09_2_layer.tif",      "Saved_Frames_A-09_2"),
    ("A-09_3", "Timing_A-09_3_layer.tif",      "Saved_Frames_A-09_3"),
    ("T-16_1", "Timing_T-16_1_layer.tif",      "Saved_Frames_T-16_1"),
    ("T-16_2", "Timing_T-16_2_layer.tif",      "Saved_Frames_T-16_2"),
    ("T-16_3", "Timing_T-16_3_layer.tif",      "Saved_Frames_T-16_3"),
    ("X5Y4_1", "Timing_red_X5Y4_1_layer.tif",  "Saved_Frames_X5Y4_1"),
    ("X5Y4_2", "Timing_red_X5Y4_2_layer.tif",  "Saved_Frames_X5Y4_2"),
    ("X5Y4_3", "Timing_red_X5Y4_3_layer.tif",  "Saved_Frames_X5Y4_3"),
]


class OCTViewer:
    """Load a 3D OCT TIFF volume, save all frames as PNGs, and provide a slider widget."""

    def __init__(self, name: str, tiff_path: str, save_dir: str):
        self.name = name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.volume = tiff.imread(tiff_path)
        self.num_frames = self.volume.shape[0]
        print(f'[{self.name}] Shape: {self.volume.shape}')

        self._save_frames()

    def _save_frames(self):
        for i in range(self.num_frames):
            path = os.path.join(self.save_dir, f'{self.name}_{i + 1}.png')
            plt.imsave(path, self.volume[i], cmap='gray')
        print(f'[{self.name}] Saved {self.num_frames} frames to {self.save_dir}')

    def show_slider(self):
        """Return an interactive ipywidgets slider for Jupyter notebooks."""
        try:
            import ipywidgets as widgets
        except ImportError:
            raise ImportError('ipywidgets is required for interactive viewing: pip install ipywidgets')

        def view(index):
            plt.figure(figsize=(5, 5))
            plt.imshow(self.volume[index], cmap='gray')
            plt.title(f'{self.name} | Frame {index + 1}/{self.num_frames}')
            plt.axis('off')
            plt.show()

        return widgets.interactive(
            view,
            index=widgets.IntSlider(0, 0, self.num_frames - 1, description=self.name[:7]),
        )


def main(args):
    tiff_dir = Path(args.tiff_dir)
    out_dir = Path(args.out_dir)

    for name, filename, out_subdir in DATASETS:
        tiff_path = tiff_dir / filename
        if not tiff_path.exists():
            print(f'[{name}] Not found: {tiff_path} — skipping')
            continue
        OCTViewer(name, str(tiff_path), str(out_dir / out_subdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract B-scan frames from 3D OCT TIFF volumes')
    parser.add_argument('--tiff-dir', required=True, help='Directory containing input .tif files')
    parser.add_argument('--out-dir',  required=True, help='Root directory to save extracted PNG frames')
    main(parser.parse_args())
