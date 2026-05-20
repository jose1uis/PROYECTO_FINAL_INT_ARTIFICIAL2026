from pathlib import Path

from src.data_utils import make_demo_dataset


def main() -> None:
    output_path = Path("data/raw/demo_student_dropout.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = make_demo_dataset()
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()

