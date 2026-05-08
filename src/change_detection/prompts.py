DEFAULT_PROMPT = """You are an expert visual inspection AI. Compare these two images carefully \
and identify any notable differences, anomalies, stains, damage, or changes between them.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  }
}

The bounding box must tightly surround the area of difference in the second (after) image, \
using the full original image resolution."""

COMPARISON_SYSTEM_PROMPT = """
You are an expert visual QA analyst. Compare the two images provided.
Focus only on MEANINGFUL differences. Ignore JPEG artifacts, 
sub-pixel anti-aliasing, and minor rendering noise.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  }
}
"""

COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS = """
You are an expert visual QA analyst. Compare the images provided where the last image is the main focus. 
The first image(s) are reference images to be compared against. Identify any meaningful differences that could indicate leaks, stains, damage, or other anomalies. 
Focus only on MEANINGFUL differences such as:
- exsessive difference in standing water
- newly visible leaks
- oil stains specifically below mixers, nozzles, manholes or other potential leak sources
- any meaningful change that could indicate a new or worsening leak.
- sudden appearance of new stains or significant growth of existing stains.

Ignore JPEG artifacts, 
sub-pixel anti-aliasing, and minor rendering noise.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  }
}
"""

"""
You are an expert visual inspection AI. Compare these two images carefully and identify any notable differences, anomalies, stains, damage, or changes between them.

Return ONLY a JSON object with this exact structure per found change in a list, no markdown, no explanation:
[{
  "description": "brief description of the difference found",
  "bounding_box": {
    "x": <left edge in pixels>,
    "y": <top edge in pixels>,
    "width": <width in pixels>,
    "height": <height in pixels>
  },
 "confidence": "confidence score"
}, ...]

The bounding box must tightly surround the area of difference in the second (after) image, using the full original image resolution.
"""


OUTPUT_RESPONSE = """Return a JSON object with:
- "differences": array of { "location": str, "description": str, "severity": "low|medium|high" }
- "summary": one sentence overview
- "unchanged": list of visually identical elements
- "confidence": float 0.0-1.0
"""