"""Vocabulary and sentence frames for generating self-correction training data.

Kept separate from the generator so the two can be varied independently: the
generator decides *how* a correction is phrased, this file decides *what* is
being talked about. Widening the banks widens the dataset without touching any
logic.

A note on register. The target input is not written English -- it is what an ASR
model emits from someone dictating: lowercase, unpunctuated, with fillers and
restarts left in. Every frame here is written to survive being lowercased and
stripped of punctuation without becoming ambiguous.
"""

import random

# --- entity banks -----------------------------------------------------------
# Each bank is a pool of interchangeable values for one slot type. Values within
# a bank must be mutually substitutable: the generator picks two at random and
# treats one as retracted and the other as chosen, so a bank containing values
# that could co-occur naturally ("Monday", "morning") would manufacture pairs
# that are not really corrections.

NAMES = [
    "john", "jane", "sarah", "michael", "priya", "omar", "susan", "mark",
    "elena", "tom", "aisha", "david", "yuki", "carlos", "nina", "raj",
    "hannah", "peter", "mei", "lucas", "fatima", "george", "olga", "sam",
    "chloe", "andre", "ingrid", "hassan", "maria", "kenji", "sofia", "liam",
]

DAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
]

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
]

TIMES = [
    "nine am", "ten am", "eleven am", "noon", "one pm", "two pm", "three pm",
    "four pm", "five pm", "six thirty", "seven fifteen", "eight forty five",
    "half past two", "quarter to four",
]

NUMBERS = [
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "twelve", "fifteen", "twenty", "twenty five", "thirty", "forty", "fifty",
    "sixty", "a hundred", "a hundred and twenty", "two hundred",
]

ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "tenth", "eleventh",
    "fifteenth", "twentieth", "twenty first", "thirtieth",
]

PLACES = [
    "berlin", "munich", "dublin", "frankfurt", "london", "paris", "madrid",
    "lisbon", "warsaw", "prague", "vienna", "oslo", "helsinki", "toronto",
    "austin", "seattle", "singapore", "sydney", "tokyo", "boston",
]

ROOMS = [
    "the main hall", "the blue room", "the annex", "meeting room two",
    "the boardroom", "the small kitchen", "the studio", "the training room",
]

TEAMS = [
    "the finance team", "the audit team", "the platform team", "marketing",
    "legal", "the design team", "accounts payable", "the support desk",
    "the data team", "the security team", "procurement", "the release team",
]

PRODUCTS = [
    "the mobile app", "the dashboard", "the billing service", "the api",
    "the onboarding flow", "the search page", "the export tool",
    "the admin panel", "the notification service", "the reporting job",
]

VERBS_TRANSITIVE = [
    ("postpone", "cancel"), ("approve", "reject"), ("merge", "revert"),
    ("archive", "delete"), ("pause", "stop"), ("rename", "duplicate"),
    ("deploy", "roll back"), ("publish", "unpublish"), ("hire", "promote"),
    ("book", "cancel"), ("forward", "archive"), ("upgrade", "downgrade"),
]

DURATIONS = [
    "two weeks", "three weeks", "a month", "two months", "ten days",
    "a fortnight", "six weeks", "a quarter", "two days", "a week",
]

MONEY = [
    "twenty pounds", "thirty pounds", "fifty euros", "sixty euros",
    "ten thousand", "twenty thousand", "two hundred dollars",
    "five hundred dollars", "a thousand euros", "fifteen hundred",
]

# --- correction signals -----------------------------------------------------
# How a speaker flags that what they just said was wrong. `full` signals read
# naturally before a complete restatement ("no wait, the meeting is on
# Saturday"); `elliptical` ones read naturally before a bare replacement ("no
# wait, Saturday"). Several work in both positions and appear in both lists.

SIGNALS_FULL = [
    "no wait", "wait no", "no sorry", "sorry", "no never mind", "never mind",
    "scratch that", "actually no", "hold on", "hang on", "strike that",
    "forget that", "my mistake", "that's wrong", "let me rephrase",
    "correction", "no hang on", "sorry no", "er no", "um no", "no actually",
    "let me start again", "no i mean", "hold on no",
]

SIGNALS_ELLIPTICAL = [
    "i mean", "i meant", "or rather", "make that", "sorry", "no wait",
    "correction", "rather", "no", "or", "make it", "i should say",
    "actually", "no sorry", "scratch that", "er", "sorry i mean",
]

# Fillers a transcriber leaves in. Weighted by how often they really appear.
#
# "I mean" is deliberately absent even though people say it constantly as a
# filler: it is also one of the strongest correction signals, and injecting it
# as noise would label the same phrase both ways within one dataset. The
# preservation frames cover its innocent use, where the surrounding words settle
# the ambiguity.
FILLERS = ["um", "uh", "er", "like", "you know", "so", "well", "kind of"]
FILLER_WEIGHTS = [22, 20, 8, 10, 8, 14, 8, 4]

OPENERS = [
    "so", "okay so", "right so", "um so", "yeah so", "okay", "right",
    "so yeah", "alright", "hey", "quick one", "just so you know",
]

# --- sentence frames --------------------------------------------------------
# `{slot}` marks the correctable value. `kind` names the bank it draws from.
# `tail` is text that must survive the correction untouched -- it is what makes
# a case test whether the model edits surgically rather than truncating, which
# is the failure mode that produced the loudest complaints.

FRAMES = [
    # scheduling
    ("the meeting is on {slot}", "day", ""),
    ("there is going to be a meeting on {slot}", "day", ""),
    ("let's meet on {slot}", "day", ""),
    ("the workshop is on {slot}", "day", "in {room}"),
    ("we're doing the review on {slot}", "day", "with {team}"),
    ("the deadline is {slot}", "day", "so we need to move"),
    ("i'll send the draft over on {slot}", "day", ""),
    ("standup moves to {slot}", "day", "this week"),
    ("can we shift the retro to {slot}", "day", ""),
    ("the release goes out {slot}", "day", "assuming qa signs off"),
    # times
    ("the call is at {slot}", "time", ""),
    ("let's start at {slot}", "time", ""),
    ("book the room for {slot}", "time", ""),
    ("the demo is at {slot}", "time", "in {room}"),
    ("i'll ring you at {slot}", "time", ""),
    # people
    ("send it to {slot}", "name", ""),
    ("ask {slot} to review it", "name", ""),
    ("invite {slot}", "name", "to the kickoff"),
    ("assign the ticket to {slot}", "name", ""),
    ("{slot} is running the session", "name", ""),
    ("loop in {slot}", "name", "before we ship"),
    ("i spoke to {slot}", "name", "about the timeline"),
    # teams
    ("send the invoice to {slot}", "team", ""),
    ("escalate it to {slot}", "team", ""),
    ("this belongs to {slot}", "team", "not us"),
    ("i'll copy {slot}", "team", "on the thread"),
    # places
    ("the office is in {slot}", "place", ""),
    ("the server is in {slot}", "place", ""),
    ("she's flying to {slot}", "place", "on sunday"),
    ("the conference is in {slot}", "place", "this year"),
    # numbers
    ("we need {slot} chairs", "number", ""),
    ("book it for {slot} people", "number", ""),
    ("we need {slot} licences", "number", "for the new starters"),
    ("there were {slot} tickets in the queue", "number", ""),
    ("order {slot} laptops", "number", "for the new team"),
    ("we should hire {slot} engineers", "number", "this quarter"),
    # money
    ("the total is {slot}", "money", ""),
    ("it costs {slot}", "money", ""),
    ("the budget is {slot}", "money", "for the whole year"),
    ("they quoted us {slot}", "money", ""),
    # months / durations / ordinals
    ("we ship in {slot}", "month", ""),
    ("the audit is in {slot}", "month", "so we have time"),
    ("it takes {slot}", "duration", ""),
    ("the contract runs for {slot}", "duration", ""),
    ("the deadline is the {slot}", "ordinal", ""),
    ("we're on the {slot} floor", "ordinal", ""),
    # products
    ("the bug is in {slot}", "product", ""),
    ("we're rewriting {slot}", "product", "next sprint"),
    ("{slot} is down again", "product", ""),
    # rooms
    ("we're in {slot}", "room", ""),
    ("the interview is in {slot}", "room", "at two"),
]

# Frames whose correctable slot is a verb, so the replacement is an action
# rather than an entity. These are here because a model that learns "corrections
# swap nouns" fails on them, and that was a real observed failure.
VERB_FRAMES = [
    "we should {slot} the launch",
    "let's {slot} the contract",
    "i want to {slot} the release",
    "we're going to {slot} the campaign",
    "can you {slot} the order",
    "they decided to {slot} the project",
    "i'd like to {slot} the booking",
]

BANKS = {
    "name": NAMES,
    "day": DAYS,
    "month": MONTHS,
    "time": TIMES,
    "number": NUMBERS,
    "ordinal": ORDINALS,
    "place": PLACES,
    "room": ROOMS,
    "team": TEAMS,
    "product": PRODUCTS,
    "duration": DURATIONS,
    "money": MONEY,
}

# --- preservation frames ----------------------------------------------------
# Sentences that contain correction vocabulary used innocently, or two values
# that are both real. The model must leave the meaning of these alone.
#
# These carry as much weight as the corrections. A model that cuts here is
# destroying dictation rather than tidying it, which is strictly worse than
# doing nothing -- so the generator is deliberately biased to produce a lot of
# them.

PRESERVE_FRAMES = [
    # signal words used as ordinary vocabulary
    "sorry i am late the meeting is on {day}",
    "sorry about the noise i'm on the train",
    "i mean what i say and i say what i mean",
    "i know what you mean about the deadline being tight",
    "wait for the build to finish before you deploy",
    "we should wait until {day} before deciding anything",
    "he said no wait for me and then hung up",
    "the correction was published in the journal last week",
    "please correct the invoice and resend it on {day}",
    "that is actually really good news for the whole team",
    "actually i think this is the better approach",
    "no problem the meeting is still on {day}",
    "turn right at the lights and the office is on the left",
    "hold on to the receipts for the expense claim",
    "she never minds working late on a release",
    "scratch the surface and it's the same problem",
    "my mistake was not testing it on staging first",
    "i'd rather we shipped it on {day}",
    "make that your priority for the week",
    "strike action is affecting the {place} office",
    # two values that are both real
    "the workshop runs on {day} and {day2}",
    "invite both {name} and {name2} to the review",
    "we can meet on {day} or {day2} whichever suits you",
    "we need between {number} and {number2} people",
    "the budget is between {money} and {money2}",
    "the design review is on {day} and the launch is on {day2}",
    "{name} is presenting and {name2} is taking notes",
    "we ship in {month} and review in {month2}",
    "send it to {name} and copy {name2}",
    "the call is at {time} and the demo is at {time2}",
    # negation and contrast that must survive
    "we should not merge this until the tests pass",
    "it is slower than the old one but far more reliable",
    "don't send it to {name} until legal has signed off",
    "this isn't ready and i don't want to rush it",
    "not everyone can make {day} so let's record it",
    # nothing to do at all
    "the deployment finished successfully this morning",
    "what time does the {place} office open on weekdays",
    "i've pushed the branch and opened a pull request",
    "the invoice has been paid and reconciled",
    "can you review the design doc before {day}",
    "we need to book the room order lunch and print the agenda",
    "thanks for turning that around so quickly",
    "the numbers look right to me now",
    "i'll write it up and share it with {team}",
    "let me know if {time} works for you",
]


def weighted_filler(rng: random.Random) -> str:
    """A filler word, drawn to roughly match how often each really occurs."""
    return rng.choices(FILLERS, weights=FILLER_WEIGHTS, k=1)[0]


def two_distinct(rng: random.Random, bank: list) -> tuple:
    """Two different values from a bank, for the retracted/chosen pair."""
    a, b = rng.sample(bank, 2)
    return a, b
