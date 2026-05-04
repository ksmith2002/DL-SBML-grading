from pathlib import Path
import numpy as np
import pydicom
from PIL import Image


# =========================
# CONFIGURE THESE PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "mri_dicoms"  # Change this to your DICOM directory
OUTPUT_DIR = BASE_DIR / "mri_pngs"  # Change this to your desired output directory


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize image data to 0-255 uint8 for PNG export.
    Uses min-max normalization.
    """
    img = img.astype(np.float32)

    img_min = np.min(img)
    img_max = np.max(img)

    if img_max == img_min:
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - img_min) / (img_max - img_min)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def dicom_to_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Extract pixel data from a DICOM dataset and apply common adjustments.
    """
    img = ds.pixel_array.astype(np.float32)

    ##################################################
    # CHAT GPT suggested these DICOM preprocessing / intensity corrections below, but they can be very dataset-specific. 
    # i tested it out and it made Rutaraj's MRI's look worse, so i commented it out for now. 
    ##################################################

    # # Apply rescale slope/intercept if present
    # slope = float(getattr(ds, "RescaleSlope", 1))
    # intercept = float(getattr(ds, "RescaleIntercept", 0))
    # img = img * slope + intercept

    # # If MONOCHROME1, invert so brighter values appear naturally
    # photometric = getattr(ds, "PhotometricInterpretation", "")
    # if photometric == "MONOCHROME1":
    #     img = np.max(img) - img

    return img


def save_png(img_array: np.ndarray, out_path: Path):
    """
    Save a 2D numpy array as a PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_array)
    img.save(out_path)


def process_dicom_file(dcm_path: Path, input_root: Path, output_root: Path):
    """
    Convert one DICOM file to PNG.
    If multi-frame, save each frame separately.
    """
    try:
        ds = pydicom.dcmread(dcm_path)

        if "PixelData" not in ds:
            print(f"Skipping (no pixel data): {dcm_path}")
            return

        img = dicom_to_array(ds)

        relative_path = dcm_path.relative_to(input_root)
        base_output = output_root / relative_path.with_suffix("")

        if img.ndim == 3:
            for i in range(img.shape[0]):
                frame = normalize_to_uint8(img[i])
                out_path = base_output.parent / f"{base_output.name}_frame_{i:03d}.png"
                save_png(frame, out_path)
                print(f"Saved: {out_path}")
        elif img.ndim == 2:
            img_uint8 = normalize_to_uint8(img)
            out_path = base_output.with_suffix(".png")
            save_png(img_uint8, out_path)
            print(f"Saved: {out_path}")
        else:
            print(f"Skipping (unsupported shape {img.shape}): {dcm_path}")

    except Exception as e:
        print(f"Failed on {dcm_path}: {e}")



def main():
    if not INPUT_DIR.exists():
        print(f"Input folder does not exist: {INPUT_DIR}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try every file
    all_files = [p for p in INPUT_DIR.rglob("*") if p.is_file()]

    print(f"Scanning: {INPUT_DIR}")
    print(f"Saving PNGs to: {OUTPUT_DIR}")
    print(f"Found {len(all_files)} total files\n")

    for file_path in all_files:
        process_dicom_file(file_path, INPUT_DIR, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()