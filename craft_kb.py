"""
Craft knowledge base — a quick-cache of original story-craft notes + mini-examples
the director cross-references while writing (and that doubles as training signal).

All ORIGINAL (no copyrighted text), so it's safe to ship and to train on. Each entry
is a craft principle + a tiny original demonstration. retrieve() pulls the relevant
ones for a concept so the model can make structural adjustments on the fly.
"""

import math
import re

KB = [
    # ── three-act spine ──────────────────────────────────────────────────────
    dict(id="act1", tags="act1 structure setup opening beginning",
         principle="ACT I — establish who, where, and what's normal, then break it. Open in "
                   "motion, show the hero's WANT fast, and end Act I on the inciting incident that "
                   "forces the story to begin.",
         example="A lamp-lighter loves his quiet street (normal + want: peace) — until the night a "
                 "lantern won't light and the dark starts creeping closer (inciting incident)."),
    dict(id="inciting", tags="act1 inciting incident spark hook",
         principle="The inciting incident is the irreversible event that ends the ordinary world. "
                   "It should arrive early and give the hero a clear, urgent problem.",
         example="The cat doesn't just appear — it drops a single key at the robot's feet and runs."),
    dict(id="act2", tags="act2 middle rising tension escalation try-fail",
         principle="ACT II — rising tension through try/fail cycles. Each attempt raises the stakes "
                   "and costs something. Avoid repetition: every beat must change the situation.",
         example="He tries the door (locked), the window (too high), the chimney (a wasp's nest) — "
                 "each fail funnier and more desperate than the last."),
    dict(id="midpoint", tags="act2 midpoint turn reversal twist",
         principle="The MIDPOINT flips the story: a reveal or reversal that changes the hero's goal "
                   "or understanding. False victory becomes real danger, or vice versa.",
         example="She reaches the treasure — and learns the map was hers all along, drawn by the "
                 "father she's been chasing."),
    dict(id="lowpoint", tags="act2 low point all-is-lost dark crisis",
         principle="Near the end of Act II, the ALL-IS-LOST: the hero's worst moment, where the "
                   "old self can't win. This earns the change to come.",
         example="The fire goes out. The cat is gone. The robot sits alone in the dark it built."),
    dict(id="climax", tags="act3 climax confrontation decision choice",
         principle="ACT III — the CLIMAX is an active CHOICE, not luck. The hero confronts the core "
                   "problem using what they learned, and pays the price or claims the reward.",
         example="He stops running, turns to face the dark, and lights the lantern with his own "
                 "trembling hand."),
    dict(id="resolution", tags="act3 ending resolution denouement final-image",
         principle="The RESOLUTION shows the new normal — proof the hero changed. End on a precise, "
                   "earned FINAL IMAGE that echoes the opening, transformed. Short. No moralizing.",
         example="The lamp-lighter walks the same street — but now he's whistling, and the dark "
                 "keeps a respectful distance."),

    # ── endings (they asked for strong endings) ──────────────────────────────
    dict(id="earned_ending", tags="ending earned satisfying payoff",
         principle="A strong ending is EARNED and INEVITABLE-YET-SURPRISING: it pays off the want "
                   "and the flaw, and recontextualizes the opening. Cut the last line you write — "
                   "the real ending is usually one beat sooner.",
         example="Not 'they lived happily ever after,' but: 'She left the porch light on. Just in "
                 "case. She always would now.'"),
    dict(id="button", tags="ending button last-beat humor twist kicker",
         principle="End many scenes/films on a BUTTON — a tiny final beat (a look, a line, a gag) "
                   "that lands the emotion or twists the knife.",
         example="The dragon pours two cups. 'One sugar or two?' Smoke curls from the knight's "
                 "abandoned helmet."),

    # ── character ─────────────────────────────────────────────────────────────
    dict(id="want_need", tags="character want need flaw goal motivation",
         principle="A hero has a WANT (external goal) and a NEED (the inner truth they're missing). "
                   "The plot chases the want; the arc delivers the need, often by sacrificing the want.",
         example="The mouse WANTS the moon; what he NEEDS is to stop being alone — so he trades the "
                 "ladder's last rung to pull a friend up beside him."),
    dict(id="obstacle", tags="character obstacle conflict stakes opposition",
         principle="Conflict = a want meeting a worthy obstacle. Stakes must be clear and personal. "
                   "No obstacle, no story.",
         example="Not 'a knight walks' but 'a knight must cross a war to deliver one flower before "
                 "it wilts.'"),

    # ── technique ──────────────────────────────────────────────────────────────
    dict(id="show", tags="show-dont-tell sensory concrete image craft",
         principle="SHOW, don't tell. Replace stated emotion with a concrete, sensory action.",
         example="Not 'he was nervous' but 'he counted the rivets on the door for the third time.'"),
    dict(id="dialogue", tags="dialogue subtext brevity voice speech",
         principle="Great dialogue is SHORT and carries SUBTEXT — characters rarely say what they "
                   "mean. One vivid line beats a speech.",
         example="'Stay.' — one word doing the work of a paragraph."),
    dict(id="humor", tags="humor comedy setup payoff incongruity timing",
         principle="Comedy = setup + surprising-but-logical payoff, or incongruity (the wrong thing "
                   "in the right place). Timing: land the joke on the last word.",
         example="The fearsome monster lurches forward, claws out — and gently offers a slightly "
                 "squashed cupcake."),
    dict(id="escalation", tags="escalation stakes raise pace momentum",
         principle="ESCALATE — each beat should raise stakes, tighten time, or deepen cost. If a "
                   "scene could be cut without loss, cut it.",
         example="A drip → a leak → ankle-deep → the candle floats away → the match is the last "
                 "dry thing left."),
    dict(id="theme", tags="theme meaning unity throughline",
         principle="THEME is the question the film keeps asking, dramatized — never stated. Every "
                   "shot should quietly serve one idea.",
         example="If the theme is 'courage is acting while afraid,' every beat tests fear, and the "
                 "win is a frightened choice — not the absence of fear."),
]


def _toks(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 3]


def retrieve(query, k=4):
    """Most-relevant craft notes for a concept/request (TF-IDF over tags+principle)."""
    qs = set(_toks(query))
    docs = [(e, set(_toks(e["tags"] + " " + e["principle"]))) for e in KB]
    df = {}
    for _, words in docs:
        for w in words:
            df[w] = df.get(w, 0) + 1
    n = len(docs)

    def score(words):
        return sum(math.log(1 + n / df.get(w, 1)) for w in qs & words)

    ranked = sorted(docs, key=lambda d: score(d[1]), reverse=True)
    return [e for e, _ in ranked[:k]]


# the always-on spine + the concept-relevant notes, compact for prompt injection
SPINE = ["act1", "inciting", "midpoint", "lowpoint", "climax", "earned_ending"]


def brief(concept):
    by_id = {e["id"]: e for e in KB}
    notes = [by_id[i] for i in SPINE]
    for e in retrieve(concept, 3):
        if e not in notes:
            notes.append(e)
    return "\n".join(f"- {e['principle']} e.g. {e['example']}" for e in notes)
