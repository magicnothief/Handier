"""Vocabulary for generating self-corrections, fillers and the negatives.

Kept apart from the generator so the two can be reviewed independently: this
file decides *what the task looks like*, and `build_dataset.py` decides how many
of each to make.

The lists are deliberately wider than `scripts/enhance-eval/suite.py`. The suite
is the exam; if training only ever saw the exam's twenty signal phrases the
score would measure recall of those phrases rather than the ability to spot a
retraction. Every phrase in the suite appears here, but so do forty more.
"""

import random
import re

# --- how a speaker signals they are taking something back -------------------
# Split by grammar, because they do not all combine with a replacement the same
# way. "make that six" is fine; "never mind six" is not.

# Stand alone before the replacement: "<X> SIGNAL <Y>"
RETRACTION_SIGNALS = [
    "no wait",
    "wait no",
    "no no wait",
    "hold on",
    "hang on",
    "sorry",
    "no sorry",
    "sorry no",
    "oops sorry",
    "scratch that",
    "strike that",
    "forget that",
    "ignore that",
    "correction",
    "my mistake",
    "my bad",
    "that's wrong",
    "that's not right",
    "actually no",
    "no actually",
    "er no",
    "um no",
    "uh no",
    "hmm no",
    "wait sorry",
    "no hang on",
    "let me rephrase",
    "let me correct that",
    "rather",
    "or rather",
    "i mean",
    "i meant",
    "i mean to say",
    "what i meant was",
    "sorry i meant",
    "no i mean",
    "well actually",
    "on second thought",
    "on second thoughts",
    "come to think of it",
    "actually",
    "no never mind",
    "never mind",
    "nope",
    "no",
]

# Take the replacement directly as an object: "<X> SIGNAL <Y>"
REPLACEMENT_SIGNALS = [
    "make that",
    "make it",
    "let's say",
    "better make that",
    "change that to",
    "sorry make that",
    "no make that",
]

# Fillers a speaker drops in without meaning anything by them.
FILLERS = [
    "um",
    "uh",
    "er",
    "ah",
    "erm",
    "hmm",
    "mm",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "basically",
    "literally",
    "actually",
    "obviously",
    "right",
    "so yeah",
    "i guess",
    "let me see",
    "let's see",
    "well",
    "okay so",
    "anyway",
]

# Fillers safe to place at the very start of an utterance.
LEADING_FILLERS = [
    "um",
    "uh",
    "er",
    "so",
    "so um",
    "okay so",
    "right so",
    "well",
    "yeah so",
    "um so",
    "uh so",
    "let me see",
    "i think",
]

# --- interchangeable entities, for building a retraction ---------------------
# Each list holds things of one kind, so a swap stays plausible: a speaker
# corrects Tuesday to Thursday, not Tuesday to Frankfurt.

DAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]
MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
NAMES = [
    "john", "jane", "sarah", "michael", "priya", "omar", "susan", "mark",
    "elena", "tom", "aisha", "carlos", "yuki", "hannah", "diego", "mei",
    "lucas", "fatima", "noah", "ingrid", "rahul", "sofia", "kwame", "lena",
    "arjun", "clara", "hugo", "nadia", "ravi", "beatriz",
]
TEAMS = [
    "the finance team", "the audit team", "the platform team", "the design team",
    "the legal team", "the support team", "the sales team", "the marketing team",
    "the data team", "the security team",
    "accounts payable", "the board", "the client", "procurement", "hr",
]
PLACES = [
    "berlin", "munich", "dublin", "frankfurt", "london", "paris", "madrid",
    "lisbon", "warsaw", "prague", "vienna", "oslo", "helsinki", "zurich",
    "toronto", "boston", "seattle", "austin", "singapore", "sydney",
]
ROOMS = [
    "the main hall", "the boardroom", "meeting room two", "the annex",
    "the coffee shop on main street", "the office", "the studio", "the lab",
    "the training room", "the atrium",
]
SMALL_NUMBERS = [
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "fifteen", "twenty", "twenty five", "thirty", "forty",
    "fifty", "sixty", "a hundred",
]
ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "tenth", "twelfth", "fifteenth", "twentieth", "thirtieth",
]
TIMES = [
    "eight am", "nine am", "ten am", "eleven am", "noon", "one pm", "two pm",
    "three pm", "four pm", "five pm", "half past nine", "quarter to three",
]
DURATIONS = [
    "two weeks", "three weeks", "a month", "two months", "six weeks",
    "ten days", "a fortnight", "two quarters", "one sprint", "three days",
]
MONEY = [
    "fifty euros", "sixty euros", "twenty pounds", "thirty pounds",
    "ten thousand", "twenty thousand", "five hundred dollars",
    "two thousand dollars", "a hundred euros", "fifteen hundred",
]
VERBS = [
    "postpone", "cancel", "reschedule", "approve", "reject", "delay",
    "publish", "archive", "merge", "revert", "escalate", "close",
    "extend", "shorten", "pause", "restart",
]
PRODUCTS = [
    "the invoice", "the quarterly report", "the roadmap", "the contract",
    "the proposal", "the deck", "the changelog", "the release notes",
    "the budget sheet", "the onboarding guide",
]

# Kind name -> pool. The generator picks a kind, then two distinct members.
ENTITY_KINDS = {
    "day": DAYS,
    "month": MONTHS,
    "name": NAMES,
    "team": TEAMS,
    "place": PLACES,
    "room": ROOMS,
    "number": SMALL_NUMBERS,
    "ordinal": ORDINALS,
    "time": TIMES,
    "duration": DURATIONS,
    "money": MONEY,
    "verb": VERBS,
    "product": PRODUCTS,
}

# --- sentence frames --------------------------------------------------------
# "{slot}" marks where the correctable entity goes. Each frame declares which
# kind it accepts so the substitution reads naturally.

FRAMES = [
    ("the meeting is on {slot}", "day"),
    ("there is going to be a meeting on {slot}", "day"),
    ("let's meet on {slot}", "day"),
    ("the workshop runs on {slot}", "day"),
    ("can we move the standup to {slot}", "day"),
    ("i'll send the update on {slot}", "day"),
    ("the deadline is in {slot}", "month"),
    ("we ship in {slot}", "month"),
    ("the contract renews in {slot}", "month"),
    ("send it to {slot}", "name"),
    ("ask {slot} to review it", "name"),
    ("assign the ticket to {slot}", "name"),
    ("invite {slot} to the review", "name"),
    ("{slot} is leading the project", "name"),
    ("loop in {slot} on this thread", "name"),
    ("forward it to {slot}", "team"),
    ("escalate this to {slot}", "team"),
    ("the office is in {slot}", "place"),
    ("the server is hosted in {slot}", "place"),
    ("the conference is in {slot}", "place"),
    ("book {slot} for the session", "room"),
    ("we're meeting at {slot}", "room"),
    ("book it for {slot} people", "number"),
    ("we need {slot} licences", "number"),
    ("order lunch for {slot}", "number"),
    ("we need {slot} chairs", "number"),
    ("the deadline is the {slot}", "ordinal"),
    ("it's on the {slot} of the month", "ordinal"),
    ("the call is at {slot}", "time"),
    ("let's start at {slot}", "time"),
    ("it takes {slot}", "duration"),
    ("the trial runs for {slot}", "duration"),
    ("the total is {slot}", "money"),
    ("the budget is {slot}", "money"),
    ("it costs {slot}", "money"),
    ("we should {slot} the launch", "verb"),
    ("i want to {slot} the release", "verb"),
    ("we need to {slot} the ticket", "verb"),
    ("please review {slot}", "product"),
    ("i've attached {slot}", "product"),
    ("can you update {slot}", "product"),
]

# Clauses appended to a frame so retractions are not always at the end of a
# short sentence. Real dictation keeps going after a correction.
TAILS = [
    "",
    "",
    " and let me know if that works",
    " before the end of the week",
    " so we have time to prepare",
    " because the client asked for it",
    " if that suits everyone",
    " and cc me on the reply",
    " once the tests pass",
    " unless something changes",
    " and i'll update the tracker",
    " in the meantime",
    " to keep things moving",
]

# Clauses prepended, so the correctable entity is not always sentence-initial.
HEADS = [
    "",
    "",
    "just checking, ",
    "quick note, ",
    "as discussed, ",
    "following up, ",
    "for the record, ",
    "heads up, ",
    "one more thing, ",
]

# --- negatives: text that must survive untouched ----------------------------
# The measured failure mode is over-eager cutting, so these are adversarial by
# construction: they use retraction vocabulary in innocent senses.

INNOCENT_SIGNAL_USES = [
    "sorry i am late the meeting is on {day}",
    "sorry to bother you but the report is due on {day}",
    "no problem the meeting is still on {day}",
    "i mean what i say and i say what i mean",
    "i know what you mean about the deadline being tight",
    "wait for the build to finish before you deploy",
    "we should wait until {day} before deciding anything",
    "he said no wait for me and then hung up the phone",
    "please correct the invoice and resend it on {day}",
    "the correction was published in the journal last week",
    "that is actually really good news for the whole team",
    "turn right at the lights and the office is on the left",
    "right so that is the plan for {day}",
    "hold on to the receipts for the audit",
    "hang on the wall decorations before the guests arrive",
    "the actually useful part is the summary at the end",
    "scratch the surface and you find the same problem",
    "she made a note of the correction in the margin",
    "never mind the cost we need it done properly",
    "i meant every word of that review",
    "strike action is planned for {month}",
    "forget the password reset i already did it",
    "my mistake was not writing it down sooner",
    "sort of works but not reliably",
    "the meeting is on {day} and the review is on {day2}",
    "we can meet on {day} or {day2} whichever suits you",
    "the workshop runs on {day} and {day2}",
    "invite both {name} and {name2} to the review",
    "we need between {num} and {num2} people for the workshop",
    "the budget is between {money} and {money2}",
    "send it to {name} and copy {name2}",
    "we should not merge this until the tests pass",
    "it is slower than the old one but far more reliable",
    "the deployment finished successfully this morning",
    "what time does the {place} office open on weekdays",
    "we need to book the room order lunch for {num} and print the agenda",
    "the design review is on {day} and the launch is on {day2}",
    "please book {room} and order coffee for {num}",
    "the invoice is due on the {ordinal} and the report on the {ordinal2}",
    "call {name} at {time} and email {team} afterwards",
]


def pick_two(rng: random.Random, pool):
    """Two distinct members of a pool, in a random order."""
    a, b = rng.sample(pool, 2)
    return a, b


# Words the target must show capitalised. The editor prompt asks the model to
# repair capitalisation, so a target that left "send it to yuki" lowercase would
# teach it to stop doing that -- the slot vocabularies are stored lowercase for
# ease of substitution, and have to be restored here.
_PROPER = {w: w.title() for w in DAYS + MONTHS + NAMES + PLACES}
_PROPER_RE = re.compile(
    r"\b(" + "|".join(sorted(_PROPER, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def polish(text: str) -> str:
    """Capitalise proper nouns and the pronoun "I" in a target."""
    out = _PROPER_RE.sub(lambda m: _PROPER[m.group(0).lower()], text)
    return re.sub(r"\bi\b", "I", out)


def sentence_case(text: str) -> str:
    """Capitalise the first letter and full-stop the end.

    The training target has to look like something a person would paste, and
    the model is being asked to repair punctuation as part of the job.
    """
    text = polish(text.strip())
    if not text:
        return text
    out = text[0].upper() + text[1:]
    if out[-1] not in ".!?":
        out += "."
    return out
