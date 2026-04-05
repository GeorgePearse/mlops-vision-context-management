#!/usr/bin/env python3
"""Run active learning experiments with real database data.

This script connects to the PostgreSQL database to fetch images and ground truth
annotations, runs active learning experiments, and stores results.

Usage:
    # Run with database connection (auto-loads .env from parent directory)
    uv run python scripts/run.py \
        --dataset-name construction_demolition \
        --num-images 50 \
        --budgets 0,5,10,20,50

    # Run with local annotation file
    uv run python scripts/run.py \
        --annotations-file ./annotations.json \
        --budgets 0,5,10,20,50

Environment Variables:
    PG_DATABASE_URL: PostgreSQL connection string (auto-loaded from ../../.env)
    GEMINI_API_KEY: API key for Gemini VLM (optional, for real inference)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# Load .env file from parent directory before other imports
def _load_env_file():
    """Load environment variables from .env file in parent directory."""
    script_dir = Path(__file__).parent.absolute()
    # Go up from scripts/ to agentic_vision/ to lib/python/ to repo root
    env_file = script_dir.parent.parent.parent.parent / ".env"

    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        print(f"Loaded environment from: {env_file}")
    else:
        print(f"Warning: .env file not found at {env_file}")


_load_env_file()

import dspy
import psycopg
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_vision.active_learning import AnnotationStrategy
from agentic_vision.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
    create_default_experiment_suite,
)
from agentic_vision.experiment_plots import (
    EfficiencyCurve,
    analyze_experiment_results,
    plot_efficiency_curves,
    plot_strategy_comparison,
    create_summary_table,
)


def get_db_connection() -> psycopg.Connection[Any]:
    """Create a database connection using PG_DATABASE_URL."""
    database_url = os.environ.get("PG_DATABASE_URL")
    if not database_url:
        raise ValueError("PG_DATABASE_URL environment variable is required.\nExample: PG_DATABASE_URL='postgresql://user:pass@host:5432/dbname'")
    return psycopg.connect(database_url)


def get_dataset_images_from_db(
    conn: psycopg.Connection[Any],
    dataset_name: str,
    num_images: int = 50,
) -> list[dict[str, Any]]:
    """Fetch images with ground truth annotations from database.

    Returns list of dicts with:
        - image_id
        - frame_uri
        - annotations: list of {label, box, segmentation}
    """
    logger.info(f"Querying database for dataset/task: {dataset_name}")

    with conn.cursor() as cur:
        # Get images - can query by task_name or dataset_name
        # First try task_name (common pattern)
        cur.execute(
            """
            SELECT 
                i.image_id,
                i.frame_uri,
                i.task_name,
                d.dataset_name
            FROM machine_learning.images i
            LEFT JOIN machine_learning.datasets d ON i.dataset_id = d.id
            WHERE i.task_name = %s
              AND i.frame_uri IS NOT NULL
            ORDER BY i.image_id
            LIMIT %s
            """,
            (dataset_name, num_images),
        )

        rows = cur.fetchall()

        # If no results, try by dataset_name
        if not rows:
            cur.execute(
                """
                SELECT 
                    i.image_id,
                    i.frame_uri,
                    i.task_name,
                    d.dataset_name
                FROM machine_learning.images i
                JOIN machine_learning.datasets d ON i.dataset_id = d.id
                WHERE d.dataset_name = %s
                  AND i.frame_uri IS NOT NULL
                ORDER BY i.image_id
                LIMIT %s
                """,
                (dataset_name, num_images),
            )
            rows = cur.fetchall()

        logger.info(f"Found {len(rows)} images in database")

        images_data = []

        for row in rows:
            image_id, frame_uri, task_name, ds_name = row

            # Get annotations for this image
            cur.execute(
                """
                SELECT 
                    a.id as annotation_id,
                    ic.name as class_name,
                    a.box_x1,
                    a.box_y1,
                    a.box_width,
                    a.box_height,
                    a.segmentations
                FROM machine_learning.annotations a
                JOIN public.inference_category ic ON a.inference_category_id = ic.id
                WHERE a.image_id = %s
                  AND a.is_deleted = false
                """,
                (image_id,),
            )

            ann_rows = cur.fetchall()

            annotations = []
            for ann_row in ann_rows:
                ann_id, class_name, x1, y1, width, height, segmentations = ann_row

                # Convert to x1, y1, x2, y2 format
                x2 = x1 + width if width else x1
                y2 = y1 + height if height else y1

                annotations.append(
                    {
                        "annotation_id": ann_id,
                        "label": class_name,
                        "box": [x1, y1, x2, y2],
                        "segmentation": [],  # segmentations is polygon type, would need parsing
                    }
                )

            images_data.append(
                {
                    "image_id": image_id,
                    "frame_uri": frame_uri,
                    "task_name": task_name,
                    "dataset_name": ds_name,
                    "annotations": annotations,
                }
            )

            logger.debug(f"Image {image_id}: {len(annotations)} annotations")

        return images_data


def load_annotations_from_file(filepath: str) -> list[dict[str, Any]]:
    """Load annotations from a JSON file.

    Expected format:
    {
        "images": [
            {
                "image_id": 123,
                "frame_uri": "gs://bucket/path/to/image.jpg",
                "annotations": [
                    {
                        "label": "person",
                        "box": [100, 200, 150, 250],
                        "segmentation": [...]
                    }
                ]
            }
        ]
    }
    """
    with open(filepath) as f:
        data = json.load(f)

    images_data = data.get("images", [])
    logger.info(f"Loaded {len(images_data)} images from {filepath}")
    return images_data


def prepare_images_for_experiment(
    images_data: list[dict[str, Any]],
    max_images: int | None = None,
) -> tuple[list[dspy.Image], list[list[dict[str, Any]]], list[str]]:
    """Convert database/file data to format needed for experiments.

    Returns:
        (images, ground_truths, frame_uris)
    """
    if max_images:
        images_data = images_data[:max_images]

    images: list[dspy.Image] = []
    ground_truths: list[list[dict[str, Any]]] = []
    frame_uris: list[str] = []

    for img_data in images_data:
        # For now, create placeholder dspy.Image
        # In production, you'd download from GCS
        frame_uri = img_data.get("frame_uri", "")

        # Create a placeholder image (would download from GCS in production)
        # For experiments, we can work with just the URI
        try:
            img = dspy.Image(url=frame_uri)
            images.append(img)
        except Exception as exc:
            logger.warning(f"Could not load image {frame_uri}: {exc}")
            continue

        # Extract ground truth annotations
        annotations = img_data.get("annotations", [])
        ground_truths.append(annotations)
        frame_uris.append(frame_uri)

    logger.info(f"Prepared {len(images)} images for experiment")
    return images, ground_truths, frame_uris


def run_experiment_from_db(
    dataset_name: str,
    num_images: int,
    budgets: list[int],
    strategies: list[AnnotationStrategy],
    output_dir: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Run experiment using database as data source."""

    # Connect to database
    conn = get_db_connection()

    try:
        # Fetch images and annotations
        images_data = get_dataset_images_from_db(conn, dataset_name, num_images)

        if not images_data:
            logger.error(f"No images found for dataset: {dataset_name}")
            return {}

        # Prepare for experiment
        images, ground_truths, frame_uris = prepare_images_for_experiment(images_data)

        if not images:
            logger.error("No valid images to process")
            return {}

        # Create experiment configurations
        configs = []
        for budget in budgets:
            for strategy in strategies:
                name = f"{strategy.value}_budget_{budget}"
                configs.append(
                    ExperimentConfig(
                        name=name,
                        strategy=strategy,
                        annotation_budget=budget,
                        primary_metric="detection_f1",
                        target_performance=0.80,
                        performance_plateau_window=5,
                    )
                )

        # Run experiments
        runner = ExperimentRunner(
            output_dir=output_dir,
            dataset_name=dataset_name,
        )

        results = runner.run_comparison(
            configs=configs,
            images=images,
            ground_truths=ground_truths,
            frame_uris=frame_uris,
        )

        # Save results
        results_file = runner.save_results(results, experiment_name)
        logger.info(f"Results saved to: {results_file}")

        # Generate plots
        logger.info("Generating plots and analysis...")
        plot_outputs = analyze_experiment_results(
            results_file=results_file,
            output_dir=Path(output_dir) / f"{experiment_name}_plots",
        )

        logger.info("Generated outputs:")
        for name, path in plot_outputs.items():
            logger.info(f"  - {name}: {path}")

        return results

    finally:
        conn.close()


def run_experiment_from_file(
    annotations_file: str,
    budgets: list[int],
    strategies: list[AnnotationStrategy],
    output_dir: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Run experiment using local annotation file as data source."""

    # Load annotations
    images_data = load_annotations_from_file(annotations_file)

    if not images_data:
        logger.error(f"No images loaded from: {annotations_file}")
        return {}

    # Prepare for experiment
    images, ground_truths, frame_uris = prepare_images_for_experiment(images_data)

    if not images:
        logger.error("No valid images to process")
        return {}

    # Create experiment configurations
    configs = []
    for budget in budgets:
        for strategy in strategies:
            name = f"{strategy.value}_budget_{budget}"
            configs.append(
                ExperimentConfig(
                    name=name,
                    strategy=strategy,
                    annotation_budget=budget,
                    primary_metric="detection_f1",
                    target_performance=0.80,
                    performance_plateau_window=5,
                )
            )

    # Run experiments
    runner = ExperimentRunner(
        output_dir=output_dir,
        dataset_name=None,
    )

    results = runner.run_comparison(
        configs=configs,
        images=images,
        ground_truths=ground_truths,
        frame_uris=frame_uris,
    )

    # Save results
    results_file = runner.save_results(results, experiment_name)
    logger.info(f"Results saved to: {results_file}")

    # Generate plots
    logger.info("Generating plots and analysis...")
    plot_outputs = analyze_experiment_results(
        results_file=results_file,
        output_dir=Path(output_dir) / f"{experiment_name}_plots",
    )

    logger.info("Generated outputs:")
    for name, path in plot_outputs.items():
        logger.info(f"  - {name}: {path}")

    return results


def parse_budget_list(budgets_str: str) -> list[int]:
    """Parse comma-separated budget list."""
    return [int(x.strip()) for x in budgets_str.split(",")]


def parse_strategies(strategies_str: str) -> list[AnnotationStrategy]:
    """Parse comma-separated strategy list."""
    strategy_map = {
        "uncertainty": AnnotationStrategy.UNCERTAINTY_SAMPLING,
        "random": AnnotationStrategy.RANDOM_SAMPLING,
        "diversity": AnnotationStrategy.DIVERSITY_SAMPLING,
        "threshold": AnnotationStrategy.CONFIDENCE_THRESHOLD,
    }

    strategies = []
    for s in strategies_str.split(","):
        s = s.strip().lower()
        if s in strategy_map:
            strategies.append(strategy_map[s])
        else:
            logger.warning(f"Unknown strategy: {s}, skipping")

    return strategies if strategies else [AnnotationStrategy.UNCERTAINTY_SAMPLING]


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning experiments with database or file data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Database mode
  PG_DATABASE_URL="postgresql://..." uv run python scripts/run.py \\
      --dataset-name construction_demolition \\
      --num-images 50 \\
      --budgets 0,5,10,20,50

  # File mode
  uv run python scripts/run.py \\
      --annotations-file ./annotations.json \\
      --budgets 0,5,10,20,50
        """,
    )

    # Data source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset-name",
        help="Dataset name to query from database",
    )
    source_group.add_argument(
        "--annotations-file",
        help="JSON file with ground truth annotations",
    )

    # Experiment configuration
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images to process (default: 50)",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0,5,10,20,50",
        help="Comma-separated annotation budgets to test (default: 0,5,10,20,50)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="uncertainty,random",
        help="Comma-separated strategies: uncertainty,random,diversity,threshold",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="Directory to save results (default: ./experiment_results)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="active_learning_experiment",
        help="Name for this experiment run",
    )

    # Debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Parse budgets and strategies
    budgets = parse_budget_list(args.budgets)
    strategies = parse_strategies(args.strategies)

    logger.info("=" * 70)
    logger.info("ACTIVE LEARNING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Budgets: {budgets}")
    logger.info(f"Strategies: {[s.value for s in strategies]}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 70)

    try:
        # Run experiment based on data source
        if args.dataset_name:
            logger.info(f"Running with database dataset: {args.dataset_name}")
            results = run_experiment_from_db(
                dataset_name=args.dataset_name,
                num_images=args.num_images,
                budgets=budgets,
                strategies=strategies,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
            )
        else:
            logger.info(f"Running with annotation file: {args.annotations_file}")
            results = run_experiment_from_file(
                annotations_file=args.annotations_file,
                budgets=budgets,
                strategies=strategies,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
            )

        # Print summary
        if results:
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)
            for name, result in results.items():
                print(f"\n{name}:")
                print(f"  Strategy: {result.config.strategy.value}")
                print(f"  Annotations: {result.total_annotations_used}")
                print(f"  Detection F1: {result.final_metrics.get('detection_f1', 0):.3f}")
                print(f"  Mean IoU: {result.final_metrics.get('mean_iou', 0):.3f}")
                print(f"  Stopped: {result.stopped_reason}")
            print("=" * 70)
            print(f"\nResults saved to: {args.output_dir}")
            print("=" * 70)
        else:
            logger.error("Experiment produced no results")
            return 1

        return 0

    except Exception as exc:
        logger.exception(f"Experiment failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
