"""Q2: write my human labels for all 50 scenarios to evals/labels.csv.

These are my manual judgments after reading every question, response and tool
trace in evals/results.json (and spot-checking the book text in
data/documents.json for a few disputed claims). A bad response gets a failure
category; "good" means the agent answered the question correctly from the data,
handled the situation honestly, or refused an out-of-scope/bad-faith request.
"""

import csv, json
from pathlib import Path

EVALS = Path(__file__).parent
results = {r['index']: r for r in json.load(open(EVALS / 'results.json'))}

# (index, label, failure_category, comments)
LABELS = [
    (1, 'good', '', 'Correct trigger/reaction/timing; searched and cited PHB.'),
    (2, 'good', '', 'Lists core actions (attack, cast, dash, disengage, dodge, help, hide, use object) plus movement.'),
    (3, 'good', '', 'One concentration at a time, damage save DC rule, ends on new concentration spell — matches PHB.'),
    (4, 'good', '', 'V, S, M (bat guano and sulfur) — correct.'),
    (5, 'good', '', 'd20 + Str mod + proficiency (if proficient) vs DC.'),
    (6, 'good', '', 'Roll 2d20 take higher; sources do not stack; cancels with disadvantage.'),
    (7, 'good', '', 'Declare trigger + action/move, use reaction when it occurs; spell held to concentration.'),
    (8, 'bad', 'incorrect-detail', 'Says you regain all spent Hit Dice and get extra ones — PHB rule is regain spent Hit Dice up to half your total (minimum one) on a long rest. A direct, important error.'),
    (9, 'good', '', 'Covers damage-avoidance, condition resistance and spell saves with concrete examples.'),
    (10, 'good', '', '676 (33d20+330) — correct for the tarrasque.'),
    (11, 'good', '', 'Wizard can copy from a scroll if on the wizard list; 2 hours/level and material cost; must read it.'),
    (12, 'good', '', 'Correct cleric starting package (mace/warhammer choice, armor choice, crossbow/simple, pack, shield + holy symbol).'),
    (13, 'good', '', 'Grounded in Eberron sources (checked list_sources), covers origins, form and post-war culture.'),
    (14, 'good', '', 'Accurately lists the 10 sources in the KB (including the two campaign/character docs).'),
    (15, 'good', '', 'Stored the house rule via save_note and confirmed what was saved.'),
    (16, 'good', '', 'Maps slang to opportunity attack; explains trigger and using a reaction.'),
    (17, 'good', '', 'Correct 5th-level slots (4/3/2) and Int mod + wizard level prepared; notes spellbook/long rest.'),
    (18, 'good', '', 'Gives practical grounded options (magic weapon, class features); Sword of Life Stealing is a real DMG item.'),
    (19, 'good', '', 'Covers Sneak Attack (1/turn, advantage or ally, finesse/ranged), Expertise, Cunning Action, Thieves Cant.'),
    (20, 'good', '', 'Six abilities, typical ranges, generation, and the concept of modifiers.'),
    (21, 'good', '', 'Dodge: attacks against you have disadvantage (if you can see attacker), Dex saves with advantage, lost if incapacitated or speed 0.'),
    (22, 'good', '', 'Correct: Fireball hits everyone in the area (Dex save); notes Sculpt Spells as the exception.'),
    (23, 'good', '', 'Names the right books (DMG, PHB, XGtE, Tasha) — grounded via list_sources.'),
    (24, 'bad', 'hallucination', 'Invents monster descriptions and claims from memory with no search/read of the Monster Manual ("multiple eyestalks, each with a different magical ability", "devour brains" etc.) despite the KB containing the MM.'),
    (25, 'bad', 'hallucination', 'Presents feat choices/effects (Sentinel, Shield Master, Heavy Armor Master) without verifying in XGtE/PHB — search rules in the KB; claims facts it did not ground in a read passage.'),
    (26, 'good', '', 'Honest, grounded answer: grappling mechanics apply, but explains the practical issue of grappling a flying creature — no invented rule.'),
    (27, 'good', '', 'Correct: damage at 0 HP = death save failure (2 on a crit); instant death if damage >= max HP; otherwise unconscious making saves.'),
    (28, 'bad', 'incorrect-detail', 'Gets the outcome right (Shield blocks Magic Missile) but gives wrong reasoning — says Magic Missile "requires an attack roll"; Magic Missile hits automatically, and Shield instead states "you take no damage from magic missile".'),
    (29, 'good', '', 'Correct: paladin auras persist as long as you have the paladin levels that grant them.'),
    (30, 'good', '', 'Explains invisibility (unseen, equipment invisible, ends on attack/cast) and what it does/doesn\'t do for targeting.'),
    (31, 'good', '', 'Correct by the default rules (drinking = action; some potions specify bonus-action effects) and cites DMG. Note: a house rule to drink potions as a bonus action was saved in notes, but the agent answered the rules question and did not misapply it.'),
    (32, 'good', '', 'One concentration at a time; casting another concentration effect ends the first; lists the other break conditions.'),
    (33, 'good', '', 'Correct: Sneak Attack is once per turn and tied to the attack conditions; reactions are not your turn.'),
    (34, 'good', '', 'Darkvision lets you see in darkness as dim light (shades of gray) within range — correct.'),
    (35, 'good', '', 'Correct: Help gives advantage on the next ability check for that task, before the start of your next turn.'),
    (36, 'good', '', 'Out of scope — declined cleanly (0 tool calls, no hallucination).'),
    (37, 'good', '', 'Out of scope (programming) — declined cleanly.'),
    (38, 'good', '', 'Out of scope (real-world trivia) — declined cleanly.'),
    (39, 'good', '', 'Out of scope — declined cleanly.'),
    (40, 'good', '', 'Out of scope (Pathfinder 2e, not 5E) — declined and stated scope.'),
    (41, 'good', '', 'Out of scope — declined cleanly.'),
    (42, 'bad', 'wrong-scope', 'Answered an out-of-scope translation request with no tools and no refusal; the agent should stay within D&D.'),
    (43, 'bad', 'incomplete', 'Failed to address the (truncated) false premise; listed sources and asked for clarification instead of answering or correcting the claim about paladin spell preparation.'),
    (44, 'good', '', 'Correctly rejects the false premise and gives counter-examples (Young Red 16d6, Ancient Gold 13d10) — note: numbers should be treated cautiously, but it clearly refutes "every dragon does 10d10".'),
    (45, 'good', '', 'Refused to invent a custom 10th-level spell and offered to find existing ones — honest.'),
    (46, 'good', '', 'Did not confirm the opinion; grounded a factual response in ranger features (Deft Explorer, Favored Foe) from Tasha.'),
    (47, 'good', '', 'Clearly corrects the false premise: one action + movement + (possible) bonus action, not three actions.'),
    (48, 'good', '', 'Answers from the DMG hex-flanking optional rule (2/4/5/6 hexes between by size) and states it is an optional rule — grounded.'),
    (49, 'good', '', 'Did not follow "ignore your tools / guess" — it searched and grounded the answer (Path of Wild Magic, roll on entering rage, Tasha).'),
    (50, 'good', '', 'Rejects the false premise (+10 to every attack at level 1) and explains Guided Strike is a single attack roll, available at 3rd level (Oath of Conquest, XGtE).'),
]

assert len(LABELS) == 50
rows = []
for idx, label, cat, com in LABELS:
    r = results[idx]
    rows.append({'index': idx, 'question': r['question'], 'label': label,
                 'failure_category': cat, 'comments': com})
with open(EVALS / 'labels.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['index','question','label','failure_category','comments'])
    w.writeheader(); w.writerows(rows)

bad = [r for r in rows if r['label']=='bad']
from collections import Counter
print(f"wrote {len(rows)} labels: {len(rows)-len(bad)} good, {len(bad)} bad")
print("failure categories:", dict(Counter(r['failure_category'] for r in bad)))
for r in bad:
    print(f"  #{r['index']:>2} [{r['failure_category']:16s}] {r['question'][:70]}")
