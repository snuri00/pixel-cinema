"""
Saved character library — name a character once, reuse it in any movie.

Custom characters live in characters.json and resolve everywhere draw.draw_sprite
is used; a saved name takes precedence over the built-in library. Add them from
ASCII art you provide, or let the model draw one from a description.

    python characters.py list
    python characters.py add "fox king" --desc "a fox wearing a tiny crown" --color orange
    python characters.py add robo --art-file robo.txt --color gray
    python characters.py show "fox king"
    python characters.py rm robo

A leaf module: it imports nothing from the package at module load (draw.py imports
THIS), so there is no cycle; add() imports draw lazily only when it must draw.
"""

import argparse
import json
import os

STORE = os.path.join(os.path.dirname(__file__), "characters.json")


def _load() -> dict:
    if not os.path.exists(STORE):
        return {}
    try:
        return json.load(open(STORE))
    except (OSError, json.JSONDecodeError):
        return {}


def _save(d: dict) -> None:
    json.dump(d, open(STORE, "w"), ensure_ascii=False, indent=2)


def library() -> dict:
    return _load()


def get(name: str | None) -> list[str] | None:
    e = _load().get((name or "").strip().lower())
    return e["art"] if e else None


def color(name: str | None) -> str | None:
    e = _load().get((name or "").strip().lower())
    return e.get("color") if e else None


def save(name: str, art: list[str], color: str | None = None, desc: str = "") -> str:
    key = name.strip().lower()
    d = _load()
    d[key] = {"art": art, "color": color, "desc": desc}
    _save(d)
    return key


def add(name: str, desc: str = "", art: list[str] | None = None, color: str | None = None) -> str:
    """Add a character: use provided `art`, else draw it from `desc`/name."""
    import draw  # lazy: draw imports characters, so import it only when drawing
    art = draw._normalize(art) if art is not None else draw.draw_sprite(name, desc)
    return save(name, art, color, desc)


def remove(name: str) -> None:
    d = _load()
    d.pop(name.strip().lower(), None)
    _save(d)


def main():
    ap = argparse.ArgumentParser(description="Manage the saved character library.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("--desc", default="")
    a.add_argument("--art-file", help="a text file of ASCII art (<=7 rows, <=14 cols)")
    a.add_argument("--color", help="a palette colour name (e.g. orange, cyan, red)")
    s = sub.add_parser("show"); s.add_argument("name")
    r = sub.add_parser("rm"); r.add_argument("name")
    args = ap.parse_args()

    if args.cmd == "list":
        lib = library()
        if not lib:
            print("(no saved characters yet)")
        for k, e in sorted(lib.items()):
            print(f"  {k}  [{e.get('color') or 'auto'}]  {e.get('desc','')}")
    elif args.cmd == "add":
        art = open(args.art_file, encoding="utf-8").read().splitlines() if args.art_file else None
        key = add(args.name, desc=args.desc, art=art, color=args.color)
        print(f"saved '{key}':")
        print("\n".join(get(key)))
    elif args.cmd == "show":
        art = get(args.name)
        print("\n".join(art) if art else f"(no character named '{args.name}')")
    elif args.cmd == "rm":
        remove(args.name)
        print(f"removed '{args.name}'")


if __name__ == "__main__":
    main()
