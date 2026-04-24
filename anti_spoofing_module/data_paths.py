from pathlib import Path

RAW_KEY = "RAW_DATA_PATH="
TRAIN_FOLDER = "LCC_FASD_training"
VAL_FOLDER = "LCC_FASD_development"
TEST_FOLDER = "LCC_FASD_evaluation"


def get_raw_data_path_from_gitignore() -> Path:
    gitignore_path = Path(__file__).resolve().parents[1] / ".gitignore"
    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    raw_line = next((line.strip() for line in lines if line.strip().startswith(RAW_KEY)), "")
    raw_path = raw_line.removeprefix(RAW_KEY).strip()
    if not raw_path:
        raise ValueError("Missing RAW_DATA_PATH in .gitignore.")
    return Path(raw_path).expanduser().resolve()


def get_data_dirs() -> tuple[Path, Path, Path]:
    raw_data_dir = get_raw_data_path_from_gitignore()
    return (
        raw_data_dir / TRAIN_FOLDER,
        raw_data_dir / VAL_FOLDER,
        raw_data_dir / TEST_FOLDER,
    )


def main() -> None:
    train_dir, val_dir, test_dir = get_data_dirs()
    raw_data_dir = train_dir.parent

    print(f"Raw data root: {raw_data_dir}")
    print(f" train directory: {train_dir.exists()}")
    print(f" validation directory: {val_dir.exists()}")
    print(f" test directory: {test_dir.exists()}")

    if train_dir.exists():
        print(f"Train set directory: {[p.name for p in train_dir.iterdir() if p.is_dir()]}")
    if val_dir.exists():
        print(f"Validation set directory: {[p.name for p in val_dir.iterdir() if p.is_dir()]}")
    if test_dir.exists():
        print(f"Test set directory: {[p.name for p in test_dir.iterdir() if p.is_dir()]}")

if __name__ == "__main__":
    main()
