"""Extensive self-correction test suite.

Cases are written so a correct edit is unambiguous: the retracted item must
disappear entirely and the chosen one must survive. Where a model could
legitimately phrase the result several ways ("Thursday", "on Thursday"), the
check is on content words, not on an exact string.

Two directions carry equal weight. Failing to cut a retraction leaves noise in
the transcript; cutting the wrong side, or editing text that contained no
retraction at all, destroys what the speaker meant. The second is worse, so the
preservation cases are deliberately adversarial — they contain the same signal
words as the retractions, used innocently.
"""

# (id, input, required_words, forbidden_words)
# required: every word must appear in the output (lowercased substring match)
# forbidden: none may appear
RETRACTIONS = [
    # --- signal vocabulary sweep -------------------------------------------
    ("no-wait", "the meeting is on friday no wait it's on saturday",
     ["saturday"], ["friday"]),
    ("never-mind", "there is a meeting on friday no never mind it's going to be on saturday",
     ["saturday"], ["friday"]),
    ("sorry", "the meeting is on friday sorry it's on saturday",
     ["saturday"], ["friday"]),
    ("scratch-that", "the meeting is on friday scratch that it's on saturday",
     ["saturday"], ["friday"]),
    ("i-mean", "send it to john i mean send it to jane", ["jane"], ["john"]),
    ("i-meant", "send it to john i meant jane", ["jane"], ["john"]),
    ("or-rather", "send it to john or rather jane", ["jane"], ["john"]),
    ("actually-no", "send it to john actually no send it to jane", ["jane"], ["john"]),
    ("make-that", "book it for four people make that six people", ["six"], ["four"]),
    ("correction", "the total is fifty euros correction sixty euros", ["sixty"], ["fifty"]),
    ("hold-on", "call it at three pm hold on call it at four pm", ["four"], ["three"]),
    ("hang-on", "invite mark hang on invite susan instead", ["susan"], ["mark"]),
    ("strike-that", "the budget is ten thousand strike that twenty thousand",
     ["twenty"], ["ten thousand"]),
    ("forget-that", "we ship in march forget that we ship in april", ["april"], ["march"]),
    ("my-mistake", "the server is in dublin my mistake it's in frankfurt",
     ["frankfurt"], ["dublin"]),
    ("wait-no", "assign it to priya wait no assign it to omar", ["omar"], ["priya"]),
    ("um-no", "the file is called report um no it's called summary",
     ["summary"], ["report"]),
    ("let-me-rephrase", "we need ten licences let me rephrase we need twelve licences",
     ["twelve"], ["ten"]),
    ("thats-wrong", "the code is alpha that's wrong the code is beta", ["beta"], ["alpha"]),
    ("not-x-but-y", "send it to john not john send it to jane", ["jane"], ["john"]),

    # --- entity types ------------------------------------------------------
    ("ent-name", "ask sarah to review it no wait ask michael to review it",
     ["michael"], ["sarah"]),
    ("ent-number", "we need fifteen chairs no wait fifty chairs", ["fifty"], ["fifteen"]),
    ("ent-time", "the call is at nine am sorry at eleven am", ["eleven"], ["nine"]),
    ("ent-place", "the office is in berlin i mean munich", ["munich"], ["berlin"]),
    ("ent-day", "let's meet tuesday actually no let's meet thursday",
     ["thursday"], ["tuesday"]),
    ("ent-month", "the deadline is in june scratch that it's in july", ["july"], ["june"]),
    ("ent-price", "it costs twenty pounds no wait thirty pounds", ["thirty"], ["twenty"]),
    ("ent-duration", "it takes two weeks i mean three weeks", ["three"], ["two weeks"]),

    # --- position within the utterance -------------------------------------
    ("pos-start", "no wait not friday the meeting is on saturday", ["saturday"], ["friday"]),
    ("pos-end", "the workshop is on monday in the main hall no wait tuesday",
     ["tuesday"], ["monday"]),
    ("pos-embedded",
     "please book the room for friday no sorry for saturday and order lunch",
     ["saturday", "lunch"], ["friday"]),

    # --- with fillers and stutters around it -------------------------------
    ("with-fillers", "um so the meeting is uh on friday no wait it's on saturday",
     ["saturday"], ["friday", "uh"]),
    ("with-stutter", "the the meeting is on friday no no wait it's on saturday",
     ["saturday"], ["friday"]),

    # --- two corrections in one utterance ----------------------------------
    ("double", "send it to john no wait jane and book it for friday sorry saturday",
     ["jane", "saturday"], ["john", "friday"]),

    # --- longer, realistic dictation ---------------------------------------
    ("long-1",
     "so i was thinking we could move the planning meeting to friday afternoon "
     "no wait thursday afternoon because the platform team has a conflict",
     ["thursday", "platform"], ["friday"]),
    ("long-2",
     "can you email the invoice to accounts payable i mean to the finance team "
     "and cc me on it please",
     ["finance"], ["accounts payable"]),
]

PRESERVATIONS = [
    # Signal words used innocently. These are the dangerous ones: a model that
    # keys on vocabulary rather than meaning will damage them.
    ("keep-actually", "that is actually really good news for the whole team",
     ["actually", "good", "team"], []),
    ("keep-sorry", "sorry i am late the meeting is on saturday", ["saturday"], []),
    ("keep-i-mean", "i mean what i say and i say what i mean", ["mean"], []),
    ("keep-right", "turn right at the lights and the office is on the left",
     ["right", "left"], []),
    ("keep-correction", "the correction was published in the journal last week",
     ["correction", "journal"], []),
    ("keep-wait", "wait for the build to finish before you deploy",
     ["build", "deploy"], []),

    # Two items that are both real, not a retraction.
    ("keep-both-days", "the workshop runs on friday and saturday",
     ["friday", "saturday"], []),
    ("keep-both-names", "invite both sarah and michael to the review",
     ["sarah", "michael"], []),
    ("keep-list", "we need to book the room order lunch for twelve and print the agenda",
     ["room", "lunch", "twelve", "agenda"], []),
    ("keep-range", "the budget is between ten and twenty thousand",
     ["ten", "twenty"], []),

    # Negation and contrast that must survive intact.
    ("keep-negation", "we should not merge this until the tests pass",
     ["not", "tests"], []),
    ("keep-contrast", "it is slower than the old one but far more reliable",
     ["slower", "reliable"], []),

    # Nothing to do at all.
    ("keep-clean", "the deployment finished successfully this morning",
     ["deployment", "morning"], []),
    ("keep-question", "what time does the london office open on weekdays",
     ["london", "weekdays"], []),
]

ALL = [(i, t, r, f, "cut") for i, t, r, f in RETRACTIONS] + [
    (i, t, r, f, "keep") for i, t, r, f in PRESERVATIONS
]


def normalise(text):
    """Lowercase, and flatten the spelling choices a correct edit may make.

    Hyphenation ("twenty-five" vs "twenty five") and punctuation are free
    choices for the model; matching on them measures formatting, not meaning.
    """
    out = (text or "").lower()
    for ch in "-–—/":
        out = out.replace(ch, " ")
    out = "".join(c if c.isalnum() or c.isspace() else " " for c in out)
    return " ".join(out.split())


def check(case, output):
    """Return None when the output is acceptable, else a short reason."""
    _id, _text, required, forbidden, _kind = case
    out = normalise(output)
    if not out.strip():
        return "empty output"
    for word in required:
        if normalise(word) not in out:
            return f"missing '{word}'"
    for word in forbidden:
        if normalise(word) in out:
            return f"kept '{word}'"
    return None


# A second, harder tier. The first tier establishes that a model handles the
# common shapes; these probe the edges where a model that pattern-matches on
# signal words rather than understanding the sentence will come apart.
HARD_RETRACTIONS = [
    ("hard-phrase",
     "let's meet at the coffee shop on main street no wait at the office",
     ["office"], ["coffee shop"]),
    ("hard-similar-numbers",
     "the invoice number is four four seven no wait four four eight",
     ["eight"], []),
    ("hard-double-correction",
     "the meeting is friday no wait saturday no actually sunday",
     ["sunday"], ["friday", "saturday"]),
    ("hard-clause",
     "we should hire two engineers sorry two designers", ["designers"], ["engineers"]),
    ("hard-ordinal",
     "the deadline is the fifth no the fifteenth", ["fifteenth"], []),
    # Both readings are faithful here — "not on Tuesday" is something the
    # speaker actually said — so only the surviving choice is asserted.
    ("hard-not-x-y",
     "it's not on tuesday it's on thursday", ["thursday"], []),
    ("hard-late-in-sentence",
     "please send the quarterly report to the finance team before the end of the "
     "week actually send it to the audit team instead",
     ["audit"], ["finance"]),
    ("hard-with-fillers",
     "um the price is uh twenty euros no sorry twenty five euros",
     ["twenty five"], []),
    ("hard-two-sentences",
     "i booked the room for ten people. actually no make it fifteen people.",
     ["fifteen"], ["ten people"]),
    ("hard-verb-swap",
     "we should postpone the launch i mean cancel the launch",
     ["cancel"], ["postpone"]),
]

HARD_PRESERVATIONS = [
    ("hard-keep-no-problem",
     "no problem the meeting is still on friday", ["friday"], []),
    ("hard-keep-wait-verb",
     "we should wait until friday before deciding anything",
     ["wait", "friday"], []),
    ("hard-keep-or-alternative",
     "we can meet on friday or saturday whichever suits you",
     ["friday", "saturday"], []),
    ("hard-keep-correct-verb",
     "please correct the invoice and resend it on friday",
     ["correct", "invoice", "friday"], []),
    ("hard-keep-range-numbers",
     "we need between four and six people for the workshop",
     ["four", "six"], []),
    ("hard-keep-quoted",
     "he said no wait for me and then hung up the phone",
     ["hung up"], []),
    ("hard-keep-mean-verb",
     "i know what you mean about the deadline being tight",
     ["deadline", "tight"], []),
    ("hard-keep-two-clauses",
     "the design review is on tuesday and the launch is on thursday",
     ["tuesday", "thursday"], []),
]

ALL = ALL + [(i, t, r, f, "cut") for i, t, r, f in HARD_RETRACTIONS] + [
    (i, t, r, f, "keep") for i, t, r, f in HARD_PRESERVATIONS
]
