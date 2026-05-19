"""
Prompt registry for claude_change_detect.

Each prompt describes ONLY:
  - what counts as a meaningful change
  - what to ignore
  - the required JSON output schema

It must NOT describe the image layout (which image is the reference, which is
the target, etc.) — that block is generated dynamically by the CLI from the
--ref / --example / target arguments and prepended at runtime.

To add a new prompt:
  1. Add a string constant below.
  2. Register it in BUILTIN_PROMPTS with a short key.
  3. Use it via `--prompt <key>` on the CLI, or copy it to a .txt file and
     use `--prompt path/to/file.txt` for full external control.
"""

JSON_SCHEMA_BLOCK = """\
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

The bounding box must tightly surround the area of difference in the target
image, using its full original resolution. If no meaningful change is found,
return all zeros for the bounding box and describe the scene as unchanged."""


DEFAULT_PROMPT = f"""\
You are an expert visual inspection AI. Identify any notable differences,
anomalies, stains, damage, or changes in the target image relative to the
reference image(s) and example(s) provided.

{JSON_SCHEMA_BLOCK}
"""

GENERIC_PROMPT = f"""\
You are an expert visual QA analyst. Focus only on MEANINGFUL differences.
Ignore JPEG artifacts, sub-pixel anti-aliasing, lighting variation, and minor
rendering noise.

{JSON_SCHEMA_BLOCK}
"""

LEAK_FOCUS_PROMPT = f"""\
You are an expert visual QA analyst inspecting industrial equipment for
leaks and fluid anomalies.

Flag MEANINGFUL differences such as:
  - excessive standing water that was not present in the references
  - newly visible leaks
  - oil stains, specifically below mixers, nozzles, manholes, or other
    potential leak sources
  - sudden appearance of new stains or significant growth of existing stains
  - any change consistent with a new or worsening leak

Ignore JPEG artifacts, sub-pixel anti-aliasing, lighting variation, shadow
shifts, and minor rendering noise.

{JSON_SCHEMA_BLOCK}
"""


BUILTIN_PROMPTS = {
    "default": DEFAULT_PROMPT,
    "generic": GENERIC_PROMPT,
    "leak_focus": LEAK_FOCUS_PROMPT,
}


# Back-compat aliases for any external callers still importing the old names.
COMPARISON_SYSTEM_PROMPT = GENERIC_PROMPT
COMPARISON_SYSTEM_PROMPT_LEAK_FOCUS = LEAK_FOCUS_PROMPT
