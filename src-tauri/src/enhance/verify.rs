//! Second-pass meaning verification.
//!
//! The mechanical guards in [`super::prompt::sanity_check`] compare *words*:
//! how many, and whether any are new. That catches deletion and invention, but
//! it is structurally blind to reordering. An observed failure on Qwen3 0.6B:
//!
//! ```text
//! in : let's ship it on tuesday actually no let's ship it on thursday
//! out: Let's ship it on Thursday actually no let's ship it on Tuesday
//! ```
//!
//! Same length, every word present in the original, meaning inverted — and it
//! reaches the user looking correct. No word-level check can catch that, so a
//! model has to read both versions and judge.
//!
//! Verification is a second inference pass, so it is not free. Two things keep
//! the cost down: it is skipped entirely when the edit could not have changed
//! meaning (see [`needs_verification`]), and it asks for a one-word answer
//! rather than a rewrite.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// What the verifier concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Meaning preserved; the edit is safe to paste.
    Same,
    /// Meaning changed; fall back to the raw transcript.
    Changed,
    /// The verifier's answer could not be read.
    ///
    /// Treated as `Same` by the caller: the mechanical guards have already
    /// passed, so an unreadable second opinion is no reason to throw away a
    /// rewrite that otherwise looks fine.
    Unclear,
}

/// When the verifier runs.
///
/// Defaults to [`VerifyMode::Off`]. The second pass was measured end to end on
/// 400 live edits — the editor's own output, judged against references — and it
/// caught 0 of the 10 bad edits while rejecting 1 good one, at roughly double
/// the latency. The mechanical guards in [`super::prompt::sanity_check`] do the
/// work that actually pays: they are why so few edits are wrong in the first place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type, Default)]
#[serde(rename_all = "lowercase")]
pub enum VerifyMode {
    /// Never verify. Fastest, and what the mechanical guards alone give you.
    #[default]
    Off,
    /// Verify only when the edit could plausibly have changed meaning.
    Auto,
    /// Verify every edit, including ones that only touched punctuation.
    Always,
}

/// Build the verification system prompt.
///
/// Asks for one word. Small models are far better at picking between two fixed
/// answers than at explaining themselves, and a one-token answer costs almost
/// nothing next to a rewrite.
///
/// An earlier version of this text was tuned to 18/18 against Qwen3 1.7B *with
/// deliberation enabled*, and that tuning did not survive the move to the
/// current default editor, which has no thinking mode. Re-measured the way the
/// app actually runs it — the real editor's own output on the 68-case suite,
/// self-verified by the resident model — it destroyed nearly a third of correct
/// edits:
///
/// | Prompt                                     | Correct kept | Corruptions caught |
/// | ------------------------------------------ | ------------ | ------------------ |
/// | always answer SAME (control)               | 47/47        | 0/8                |
/// | resolve-then-answer scaffolding            | 23/47        | 6/8                |
/// | retraction rule plus a worked CHANGED case | 28/47        | 8/8                |
/// | earlier "disallowed rules first" version    | 33/47        | 3/8                |
/// | **this text**                              | **42/47**    | **5/8**            |
/// | permissive "what did they settle on"       | 44/47        | 3/8                |
///
/// Read down the table: the catch rate barely moves while the false-reject rate
/// swings from 0 to 24. The prompt is sliding one threshold rather than making
/// the model read the pair more carefully, so there is no wording that is both
/// safe and thorough. This text is the best balance found, and it beats the
/// version it replaces on *both* axes.
///
/// Every false reject in the old version was a self-correction — the one thing
/// this layer exists to do. That is why [`needs_verification`] now sends only
/// the edits a mechanical check has flagged as risky, instead of every edit
/// that changed a content word.
pub fn build_verify_prompt() -> String {
    // Written as one literal line per prompt line. Rust's `\` continuation
    // strips the newline but the exact indentation that survives is easy to get
    // wrong, and a prompt that differs from the measured one by stray
    // whitespace is not the prompt that was measured.
    concat!(
        "A dictated transcript has been edited. Decide whether the edited version still says what the speaker meant.\n",
        "\n",
        "Speakers often change their mind mid-sentence. When they do, the edit is SUPPOSED to delete the wording they abandoned along with the phrase that signalled the change. That deletion is correct: the edited version will be shorter and the abandoned wording will be gone. Answer SAME for those.\n",
        "\n",
        "For example, \"the meeting is on friday no wait it's on saturday\" edited to \"The meeting is on Saturday.\" is SAME. Friday was abandoned, so it is meant to be gone.\n",
        "\n",
        "Answer CHANGED only when the edit got the speaker's final choice wrong:\n",
        "- it kept the wording the speaker abandoned instead of the wording they chose\n",
        "- it swapped or reordered names, dates, times or numbers\n",
        "- it reversed, negated or contradicted a statement\n",
        "- it added a claim the speaker never made\n",
        "- it dropped something the speaker never took back\n",
        "\n",
        "Reply with exactly one word: SAME or CHANGED.",
    )
    .to_string()
}

/// Build the user turn presenting both versions.
pub fn build_verify_input(original: &str, edited: &str) -> String {
    format!("Original: {}\nEdited: {}", original.trim(), edited.trim())
}

/// Read a verdict out of the verifier's reply.
///
/// Deliberately forgiving about surrounding text but strict about ambiguity: a
/// reply mentioning both words has not actually decided, and is reported as
/// `Unclear` rather than resolved by position.
pub fn parse_verdict(reply: &str) -> Verdict {
    let lowered = reply.to_lowercase();
    // If a reasoning block survived, judge only what follows it: deliberation
    // weighs up both answers, so matching against it decides nothing.
    let lowered = match lowered.split_once("</think>") {
        Some((_, after)) => after.to_string(),
        None => lowered,
    };
    let says_same = lowered.contains("same");
    let says_changed = lowered.contains("changed");

    match (says_same, says_changed) {
        (true, false) => Verdict::Same,
        (false, true) => Verdict::Changed,
        _ => Verdict::Unclear,
    }
}

/// Whether an edit is worth spending a verification pass on.
///
/// This is triage, and getting it wrong is expensive in both directions: too
/// broad and the verifier's false rejects eat the feature, too narrow and a
/// corrupted rewrite reaches the user.
///
/// An earlier version returned true for *any* content-word change, which meant
/// every self-correction was verified — and since a self-correction necessarily
/// deletes content words, that is also the shape the verifier is worst at
/// judging. Measured over the real editor's output on the 68-case suite, that
/// sent 47 edits to the verifier and lost 14 correct ones.
///
/// So instead of asking "did anything change", this asks "did something change
/// that a word-level check cannot already vouch for":
///
/// - **Reordering** ([`reorders_shared_words`]) is the blind spot the whole
///   module was built for. Words can survive a swap intact, so no count-based
///   guard can see it, but the meaning inverts.
/// - **Novel content words** mean the model wrote something that was not
///   dictated. `prompt::sanity_check` already rejects wholesale invention; this
///   catches the smaller case that slips under that threshold.
///
/// An order-preserving deletion is left alone. That is the ordinary retraction,
/// the editor handles it at 100% on the suite, and it is exactly what the
/// verifier used to throw away. Of 47 real edits only 4 are now referred, none
/// of which the verifier rejects — while the swap and invention cases still get
/// checked. Users who want the broad sweep anyway can choose [`VerifyMode::Always`].
pub fn needs_verification(original: &str, edited: &str) -> bool {
    reorders_shared_words(original, edited) || adds_content_words(original, edited)
}

/// Whether the edit introduced content words that were never dictated.
///
/// Compared as a set, not a sequence: a word the speaker used once and the
/// editor kept is not novel no matter where it ends up.
fn adds_content_words(original: &str, edited: &str) -> bool {
    let before: std::collections::HashSet<String> =
        content_sequence(original).into_iter().collect();
    content_sequence(edited).iter().any(|w| !before.contains(w))
}

/// Lowercased content words in order, ignoring punctuation and short function
/// words that punctuation repair legitimately introduces.
fn content_sequence(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.chars().count() >= 4)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Whether the edit reordered content words that both versions share.
///
/// This is the signal the word-level guards miss entirely. Words are compared
/// as a sequence rather than a set, so "Tuesday … Thursday" becoming
/// "Thursday … Tuesday" is visible even though both versions use both words.
pub fn reorders_shared_words(original: &str, edited: &str) -> bool {
    let before = content_sequence(original);
    let after = content_sequence(edited);

    // Only words that survive in the same multiplicity can be compared for
    // order; a word the edit legitimately dropped says nothing about ordering.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for w in &before {
        *counts.entry(w.as_str()).or_default() += 1;
    }
    let mut after_counts: HashMap<&str, usize> = HashMap::new();
    for w in &after {
        *after_counts.entry(w.as_str()).or_default() += 1;
    }

    let shared: Vec<&String> = before
        .iter()
        .filter(|w| after_counts.get(w.as_str()) == counts.get(w.as_str()))
        .collect();
    let shared_after: Vec<&String> = after
        .iter()
        .filter(|w| after_counts.get(w.as_str()) == counts.get(w.as_str()))
        .collect();

    shared != shared_after
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prints the verification prompt for the smoke-test harness, so the model
    /// sees exactly what the app sends. Run with `-- --nocapture`.
    #[test]
    fn dump_verify_prompt() {
        println!("<<<PROMPT_START>>>");
        println!("{}", build_verify_prompt());
        println!("<<<PROMPT_END>>>");
    }

    #[test]
    fn parses_a_clean_verdict() {
        assert_eq!(parse_verdict("SAME"), Verdict::Same);
        assert_eq!(parse_verdict("CHANGED"), Verdict::Changed);
        assert_eq!(parse_verdict("  same  "), Verdict::Same);
    }

    #[test]
    fn parses_a_verdict_wrapped_in_chatter() {
        assert_eq!(parse_verdict("The meaning is the SAME."), Verdict::Same);
        assert_eq!(
            parse_verdict("This has CHANGED the meaning."),
            Verdict::Changed
        );
    }

    #[test]
    fn refuses_to_guess_when_both_words_appear() {
        // A model restating the instructions has not decided anything.
        assert_eq!(
            parse_verdict("Answer SAME if preserved or CHANGED if not"),
            Verdict::Unclear
        );
    }

    #[test]
    fn reports_unreadable_replies_as_unclear() {
        assert_eq!(parse_verdict(""), Verdict::Unclear);
        assert_eq!(parse_verdict("I think it is fine"), Verdict::Unclear);
    }

    #[test]
    fn cosmetic_edits_skip_verification() {
        // Punctuation and capitalisation only.
        assert!(!needs_verification(
            "the meeting is friday we should prepare the roadmap slides",
            "The meeting is Friday. We should prepare the roadmap slides."
        ));
    }

    #[test]
    fn filler_removal_alone_skips_verification() {
        assert!(!needs_verification(
            "um so the meeting is uh moved to friday at three",
            "So the meeting is moved to Friday at three."
        ));
    }

    #[test]
    fn an_ordinary_retraction_is_not_sent_to_the_verifier() {
        // The regression this triage exists to prevent. An order-preserving
        // deletion is the ordinary self-correction; referring it cost 14 of 47
        // correct edits when measured against the real editor's output.
        assert!(!needs_verification(
            "send it to john no wait not john send it to jane instead please",
            "Send it to Jane instead please."
        ));
        // The case reported from a live session, verbatim.
        assert!(!needs_verification(
            "there is going to be a meeting on friday no never mind it's going to be on saturday",
            "There is going to be a meeting on Saturday."
        ));
    }

    #[test]
    fn a_smuggled_in_word_is_sent_to_the_verifier() {
        // "release" was never dictated. It stays under `sanity_check`'s
        // novel-word fraction *and* under its length ceiling, which is what
        // makes it the verifier's job rather than a mechanical guard's.
        assert!(needs_verification(
            "we should merge this after the tests pass on the build server",
            "We should merge this after the release tests pass on the server."
        ));
    }

    #[test]
    fn a_swap_is_still_sent_to_the_verifier() {
        // The blind spot the module was built for: every word survives, so no
        // count-based guard sees it.
        assert!(needs_verification(
            "let's ship it on tuesday actually no let's ship it on thursday so qa has time",
            "Let's ship it on Thursday actually no let's ship it on Tuesday so QA has time."
        ));
    }

    #[test]
    fn detects_the_observed_date_swap() {
        // The failure that motivated this module. Word-level guards pass it;
        // the ordering check does not.
        let original =
            "let's ship it on tuesday actually no let's ship it on thursday so qa has time";
        let edited =
            "Let's ship it on Thursday actually no let's ship it on Tuesday so QA has time.";
        assert!(
            reorders_shared_words(original, edited),
            "a swap of two shared words must register as reordering"
        );
        assert!(needs_verification(original, edited));
    }

    #[test]
    fn faithful_edit_is_not_flagged_as_reordering() {
        let original = "um so the meeting is uh moved to friday at three";
        let edited = "So the meeting is moved to Friday at three.";
        assert!(!reorders_shared_words(original, edited));
    }

    #[test]
    fn verify_mode_defaults_to_off() {
        // Measured on 400 live edits the pass caught 0 of 10 bad ones and
        // rejected a good one, so it is opt-in rather than opt-out.
        assert_eq!(VerifyMode::default(), VerifyMode::Off);
    }

    #[test]
    fn prompt_names_both_answers_and_the_swap_rule() {
        let p = build_verify_prompt();
        assert!(p.contains("SAME"));
        assert!(p.contains("CHANGED"));
        assert!(p.contains("swapped or reordered"));
    }

    #[test]
    fn prompt_states_that_a_retraction_deletion_is_correct() {
        // The single change worth the most: without it the model reads every
        // retraction as dropped information and answers CHANGED. Measured at
        // 33/47 correct edits kept before, 42/47 after.
        let p = build_verify_prompt();
        assert!(p.contains("SUPPOSED to delete the wording they abandoned"));
        assert!(
            p.contains("the edited version will be shorter"),
            "the model has to be told a shorter result is expected"
        );
    }

    #[test]
    fn prompt_works_a_retraction_example_through_to_same() {
        // The abstract rule alone did not land; a worked instance did.
        let p = build_verify_prompt();
        let example = p
            .find("The meeting is on Saturday.")
            .expect("worked example present");
        assert!(
            p[example..].starts_with("The meeting is on Saturday.\" is SAME"),
            "the worked example must resolve to SAME"
        );
    }

    #[test]
    fn verdict_ignores_a_surviving_reasoning_block() {
        let reply = "<think>It could be SAME, but the date moved so it is CHANGED</think>
CHANGED";
        assert_eq!(parse_verdict(reply), Verdict::Changed);
        let reply = "<think>Weighing CHANGED against SAME here</think> SAME";
        assert_eq!(parse_verdict(reply), Verdict::Same);
    }

    #[test]
    fn input_presents_both_versions() {
        let s = build_verify_input("  raw text ", " edited text ");
        assert!(s.contains("Original: raw text"));
        assert!(s.contains("Edited: edited text"));
    }
}
