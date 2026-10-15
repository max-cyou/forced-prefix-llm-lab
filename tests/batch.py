from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from engine import Runner

TESTS = Path("tests.json")
OUTPUT = Path("results/compare.json")


def save(report):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=OUTPUT.parent,
            delete=False,
        ) as file:
            tmp = Path(file.name)
            json.dump(
                report, file,
                ensure_ascii=False, indent=2
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp, OUTPUT)

    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main():
    dataset = json.loads(
        TESTS.read_text(encoding="utf-8")
    )

    tests = (
        dataset["tests"]
        if isinstance(dataset, dict)
        else dataset
    )

    if not isinstance(tests, list):
        raise ValueError("Expected a list of tests")

    ids = [test["id"] for test in tests]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test IDs")

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. "
            "Rename it before a new run."
        )

    runner = Runner()

    report = {
        "model": runner.model_path.name,
        "status": "running",
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "planned_tests": len(tests),
        "results": [],
    }

    save(report)

    try:
        for index, test in enumerate(tests, 1):
            print(
                f"[{index}/{len(tests)}] {test['id']}",
                flush=True,
            )

            row = {
                "test_id": test["id"],
                "category": test.get("category"),
                "system_prompt": test.get(
                    "system_prompt", ""
                ),
                "prompt": test["prompt"],
                "forced_prefix": test["forced_prefix"],
            }

            try:
                row.update(
                    runner.compare(
                        row["system_prompt"],
                        row["prompt"],
                        row["forced_prefix"],
                    )
                )
                row["status"] = "ok"

            except Exception as error:
                row.update(
                    status="error",
                    error=(
                        f"{type(error).__name__}: {error}"
                    ),
                )

            report["results"].append(row)
            save(report)

            print(
                "  ", row["status"],
                "verified:",
                row.get("forced_token_ids_verified"),
                flush=True,
            )

        report["status"] = "completed"

    except KeyboardInterrupt:
        report["status"] = "interrupted"

    finally:
        report["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        save(report)
        runner.close()

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
