"""
Shared type definitions for LLM Cinema: the movie spec/shot records, the render
grid cell, and the floor/scenery tables — one documented contract in one place.

A leaf module (imports only `typing`), so every other module can import it freely
without an import cycle.
"""

from typing import TypedDict

RGB = tuple[int, int, int]            # an (r, g, b) colour
Sprite = list[str]                    # ASCII art: one string per row
Cell = list                          # a grid cell: [char: str, colour: RGB]
Grid = list[list[Cell]]              # grid[y][x] -> Cell


class Shot(TypedDict, total=False):
    """One beat of a film. Keys are optional because raw model output may omit
    them; `movies.direct` fills them all in for the sanitized contract."""
    narration: str
    cast: list[str]
    action: str                       # one of movies.ACTIONS
    setting: str                      # one of movies.SETTINGS
    camera: str                       # one of movies.CAMERAS
    mood: list[str]                   # one feeling per cast member (movies.MOOD_EMOTE keys)
    dialogue: str


class MovieSpec(TypedDict, total=False):
    title: str
    logline: str
    shots: list[Shot]


class ScenePlan(TypedDict):
    """Set-dressing for one shot (see movies.scenery)."""
    sky: str | None                   # 'sun' / 'moon' / None
    clouds: int
    stars: int
    ground: list[tuple[str, float]]   # (sprite label, fractional x) background props


class FloorProfile(TypedDict):
    """A terrain band (see movies.FLOOR)."""
    surf: str                         # surface-row glyph pattern
    fill: str                         # fill-row glyph
    scol: RGB                         # surface colour
    fcol: RGB                         # fill colour
    tcol: str                         # curses Color name (terminal)
    anim: bool                        # shimmer (shift the surface each frame)
