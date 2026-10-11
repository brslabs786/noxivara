#!/bin/bash
# quick_restore.sh — One-line restore + rate workflow
# Usage: ./quick_restore.sh my_photo.jpg
# After running, it shows you the result and asks for a rating

INPUT="$1"
if [ -z "$INPUT" ]; then
  echo "Usage: ./quick_restore.sh <input_image>"
  exit 1
fi

BASENAME="${INPUT%.*}"
OUTPUT="restored_${BASENAME##*/}.png"

echo ""
echo "🔄 Restoring: $INPUT"
python3 restore.py --input "$INPUT" --output "$OUTPUT"

echo ""
echo "✅ Done! Output: $OUTPUT"
echo "   Comparison: ${OUTPUT%.*}_comparison.png"
echo ""
echo "Please rate this restoration (1=very bad, 5=excellent):"
read -p "   Rating [1-5]: " RATING
read -p "   Note (press Enter to skip): " NOTE

if [ -n "$RATING" ]; then
  python3 feedback.py --input "$INPUT" --output "$OUTPUT" --rating "$RATING" --note "$NOTE"
  echo ""
  echo "Thank you! Run 'python3 auto_improve.py' to let the model learn from your feedback."
fi
