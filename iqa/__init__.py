"""HarnessIQA: the HarnessIR harness repurposed for image quality assessment.

Two tasks share one set of stages:

    analyze   one image  -> which artifact types are present, how severe, and
                            an overall technical quality score
    compare   two images -> which one has better technical quality

and one addition HarnessIR does not have: ``tune``, which rewrites the
analyst's guidelines from labelled data (artifact labels for ``analyze``,
human preferences for ``compare``).

Run ``python -m iqa --help`` from the repository root. See iqa/README.md.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
