"""
feedback.py  —  Continuous Learning Feedback System for DAResUNet Restoration
=============================================================================
Rate your restoration results (1-5 stars). Bad ratings are saved so the model
can automatically re-tune its ISP parameters and learn from mistakes.

Usage:
    # Rate a restoration after running restore.py
    python feedback.py --input my_photo.jpg --output restored_output.png --rating 4

    # Rate with a note explaining what's wrong
    python feedback.py --input my_photo.jpg --output restored_output.png --rating 2 \\
        --note "too yellow, walls look unnatural"

    # View all stored feedback
    python feedback.py --view

    # View only bad ratings (1-2 stars)
    python feedback.py --view --bad-only

    # Show feedback statistics
    python feedback.py --stats

Ratings Scale:
    1  — Very bad (colors completely wrong, worse than input)
    2  — Bad (visible problems: oversaturation, noise, artifacts)
    3  — OK (acceptable but not great)
    4  — Good (natural looking, most details recovered)
    5  — Excellent (indistinguishable from real photography)
"""

import argparse
import os
import sys
import sqlite3
import json
import datetime
from pathlib import Path

# Feedback DB location (inside project folder)
FEEDBACK_DB = os.path.join(os.path.dirname(__file__), "feedback_db", "feedback.sqlite")


def _ensure_db():
    """Create the feedback database and tables if they do not exist."""
    os.makedirs(os.path.dirname(FEEDBACK_DB), exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS restoration_feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            input_path  TEXT    NOT NULL,
            output_path TEXT    NOT NULL,
            scene_type  TEXT,
            rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            note        TEXT,
            diag_json   TEXT,
            model_key   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS isp_param_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            scene_type    TEXT    NOT NULL,
            param_name    TEXT    NOT NULL,
            old_value     REAL,
            new_value     REAL,
            trigger_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(input_path, output_path, rating, note=None, scene_type=None,
                  diag=None, model_key="exp11"):
    """Save a restoration rating to the feedback database."""
    _ensure_db()
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""
        INSERT INTO restoration_feedback
            (timestamp, input_path, output_path, scene_type, rating, note, diag_json, model_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        os.path.abspath(input_path),
        os.path.abspath(output_path),
        scene_type,
        rating,
        note,
        json.dumps(diag) if diag else None,
        model_key,
    ))
    conn.commit()

    # Count how many bad samples exist now for this scene_type
    cursor = conn.execute(
        "SELECT COUNT(*) FROM restoration_feedback WHERE rating <= 2 AND scene_type = ?",
        (scene_type,)
    )
    bad_count = cursor.fetchone()[0]
    conn.close()

    stars = "★" * rating + "☆" * (5 - rating)
    print(f"[✓] Feedback saved  →  {stars}  (rating {rating}/5)")
    if note:
        print(f"    Note: {note}")
    if bad_count >= 3:
        print(f"\n[⚠] {bad_count} bad ratings for scene '{scene_type}'.")
        print(f"    Run:  python auto_improve.py --scene {scene_type}")
        print(f"    to automatically improve restoration for this scene type.\n")
    return bad_count


def view_feedback(bad_only=False, limit=30):
    """Display stored feedback records."""
    _ensure_db()
    conn = sqlite3.connect(FEEDBACK_DB)
    query = "SELECT id, timestamp, input_path, scene_type, rating, note FROM restoration_feedback"
    if bad_only:
        query += " WHERE rating <= 2"
    query += " ORDER BY timestamp DESC LIMIT ?"
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    if not rows:
        print("No feedback records found.")
        return

    print(f"\n{'ID':>4}  {'Date/Time':<20}  {'Scene':<20}  {'Rating':<8}  {'Input':<30}  Note")
    print("─" * 110)
    for row in rows:
        fid, ts, inp, scene, rating, note = row
        stars = "★" * rating + "☆" * (5 - rating)
        fname = os.path.basename(inp) if inp else "?"
        scene_str = (scene or "unknown")[:18]
        note_str = (note or "")[:35]
        print(f"{fid:>4}  {ts[:19]:<20}  {scene_str:<20}  {stars:<8}  {fname:<30}  {note_str}")


def show_stats():
    """Display feedback statistics."""
    _ensure_db()
    conn = sqlite3.connect(FEEDBACK_DB)

    total = conn.execute("SELECT COUNT(*) FROM restoration_feedback").fetchone()[0]
    if total == 0:
        print("No feedback recorded yet.")
        conn.close()
        return

    avg = conn.execute("SELECT AVG(rating) FROM restoration_feedback").fetchone()[0]
    bad = conn.execute("SELECT COUNT(*) FROM restoration_feedback WHERE rating <= 2").fetchone()[0]
    good = conn.execute("SELECT COUNT(*) FROM restoration_feedback WHERE rating >= 4").fetchone()[0]

    print(f"\n{'='*50}")
    print(f"  Restoration Feedback Statistics")
    print(f"{'='*50}")
    print(f"  Total ratings   : {total}")
    print(f"  Average rating  : {avg:.2f}/5.00")
    print(f"  Good (4-5★)    : {good} ({100*good//total}%)")
    print(f"  Bad  (1-2★)    : {bad} ({100*bad//total}%)")
    print(f"{'='*50}")

    # Per scene type breakdown
    rows = conn.execute("""
        SELECT scene_type, COUNT(*) AS n, AVG(rating) AS avg_r,
               SUM(CASE WHEN rating<=2 THEN 1 ELSE 0 END) AS bad_n
        FROM restoration_feedback
        GROUP BY scene_type
        ORDER BY avg_r ASC
    """).fetchall()

    print(f"\n  Scene-type breakdown:")
    print(f"  {'Scene':<25}  {'Count':>5}  {'Avg ★':>6}  {'Bad':>5}")
    print(f"  {'─'*50}")
    for scene_type, n, avg_r, bad_n in rows:
        scene_str = (scene_type or "unknown")[:24]
        bar = "★" * round(avg_r) + "☆" * (5 - round(avg_r))
        flag = " ← needs improvement!" if bad_n >= 3 else ""
        print(f"  {scene_str:<25}  {n:>5}  {avg_r:>5.2f}  {bad_n:>5}{flag}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Rate your DAResUNet image restoration results for continuous learning."
    )
    parser.add_argument("--input",   help="Original input image path")
    parser.add_argument("--output",  help="Restored output image path")
    parser.add_argument("--rating",  type=int, choices=[1, 2, 3, 4, 5],
                        help="Rating 1-5 (1=very bad, 5=excellent)")
    parser.add_argument("--note",    help="Optional note describing the problem")
    parser.add_argument("--scene",   help="Scene type (auto-detected if not provided)")
    parser.add_argument("--view",    action="store_true", help="View all feedback records")
    parser.add_argument("--bad-only", action="store_true", help="Show only bad ratings (1-2★)")
    parser.add_argument("--stats",   action="store_true", help="Show feedback statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.view:
        view_feedback(bad_only=args.bad_only)
    elif args.input and args.output and args.rating:
        if not os.path.exists(args.input):
            print(f"[ERROR] Input not found: {args.input}")
            sys.exit(1)
        if not os.path.exists(args.output):
            print(f"[ERROR] Output not found: {args.output}")
            sys.exit(1)

        # Try to auto-detect scene type from the diag log if not provided
        scene_type = args.scene
        if not scene_type:
            # Quick diagnosis
            try:
                sys.path.insert(0, os.path.dirname(__file__))
                import torch, torchvision.transforms as T
                from PIL import Image
                from restore import diagnose_scene
                img = Image.open(args.input).convert("RGB")
                x = T.ToTensor()(img).unsqueeze(0)
                diag = diagnose_scene(x)
                scene_type = diag["scene_type"]
                print(f"[Auto] Detected scene type: {scene_type}")
            except Exception as e:
                print(f"[Auto-detect skipped: {e}]")
                scene_type = "UNKNOWN"

        save_feedback(
            input_path=args.input,
            output_path=args.output,
            rating=args.rating,
            note=args.note,
            scene_type=scene_type,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python feedback.py --input photo.jpg --output restored.png --rating 4")
        print("  python feedback.py --input photo.jpg --output restored.png --rating 2 --note 'too yellow'")
        print("  python feedback.py --stats")
        print("  python feedback.py --view --bad-only")


if __name__ == "__main__":
    main()
