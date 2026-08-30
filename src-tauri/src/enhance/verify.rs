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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, specta::Type, Default)]
#[serde(rename_all = "lowercase")]
pub enum VerifyMode {
    /// Never verify. Fastest, and what the mechanical guards alone give you.
    Off,
    /// Verify only when the edit could plausibly have changed meaning.
    #[default]
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
/// The exact shape of this text matters more than it looks. Measured on an
/// 18-case set (10 corrupted edits that must be caught, 8 faithful edits that
/// must be kept) against Qwen3 1.7B:
///
/// | Prompt                                        | Score |
/// | --------------------------------------------- | ----- |
/// | disallowed rules first, with a worked instance | 18/18 |
/// | allowed rules first (earlier version)          | 17/18 |
/// | plus an explicit "resolve the correction" step | 16/18 |
/// | plus worked examples as well                   | 14/18 |
/// | terse, rules compressed to one line each       | 13/18 |
///
/// Adding reasoning scaffolding made it consistently *worse* — the model has a
/// thinking mode already, and extra procedure competes with it. What helped was
/// leading with what is disallowed and naming one concrete instance of the rule
/// it kept missing (keeping the wording a speaker had retracted).
///
/// This tuning does not transfer: the same wording scores 11/18 on Qwen3 0.6B,
/// below the 15/18 that the older phrasing got there. That is why the catalog
/// offers only a model this prompt was validated against as a verifier — see
/// `catalog::for_role`. Re-run the eval before adding another.
pub fn build_verify_prompt() -> String {
    // Written as one literal line per prompt line. Rust's `\` continuation
    // strips the newline but the exact indentation that survives is easy to get
    // wrong, and a prompt that differs from the measured one by stray
    // whitespace is not the prompt that scored 18/18.
    concat!(
        "You compare a dictated transcript with an edited version of it and decide whether the edit changed what the speaker meant.\n",
        "\n",
        "These edits DO change meaning:\n",
        "- keeping wording the speaker rejected. If they said \"not John, send it to Jane\", the result must be Jane, not John.\n",
        "- swapping or reordering names, dates, times or numbers\n",
        "- reversing, negating or contradicting a statement\n",
        "- adding a claim the speaker never made\n",
        "- dropping information the speaker did not retract\n",
        "\n",
        "These edits are allowed and do NOT change meaning:\n",
        "- removing filler words and hesitations\n",
        "- fixing punctuation, capitalisation and sentence boundaries\n",
        "- deleting wording the speaker retracted or corrected\n",
        "\n",
        "Reply with exactly one word: SAME if the meaning is preserved, or CHANGED if it is not.",
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
/// Verification exists to catch meaning changes, and an edit that only removed
/// filler words or repaired punctuation cannot have changed meaning — every
/// content word survives in the same order. Skipping those keeps the common
/// case at one inference pass instead of two, which on a low-end CPU is the
/// difference between a usable feature and an unusable one.
///
/// Returns true when content words were reordered, dropped, or added, since
/// those are the edits where the model had latitude to get it wrong.
pub fn needs_verification(original: &str, edited: &str) -> bool {
    let before = content_sequence(original);
    let after = content_sequence(edited);

    // Identical content words in identical order: only cosmetic changes.
    if before == after {
        return false;
    }

    // A pure deletion that preserves order is the self-correction case, which
    // is exactly what we most want checked; anything else differs too.
    true
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
    fn self_correction_is_verified() {
        // Content was deleted, so the model had room to delete the wrong part.
        assert!(needs_verification(
            "send it to john no wait not john send it to jane instead please",
            "Send it to Jane instead please."
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
    fn verify_mode_defaults_to_auto() {
        assert_eq!(VerifyMode::default(), VerifyMode::Auto);
    }

    #[test]
    fn prompt_names_both_answers_and_the_swap_rule() {
        let p = build_verify_prompt();
        assert!(p.contains("SAME"));
        assert!(p.contains("CHANGED"));
        assert!(p.contains("swapping or reordering"));
    }

    #[test]
    fn prompt_leads_with_what_is_disallowed() {
        // Measured: putting the allowed edits first cost a case. Keep the
        // disallowed list ahead of the allowed one.
        let p = build_verify_prompt();
        let disallowed = p
            .find("DO change meaning")
            .expect("disallowed section present");
        let allowed = p
            .find("do NOT change meaning")
            .expect("allowed section present");
        assert!(
            disallowed < allowed,
            "the disallowed rules must come first; reordering measurably hurt accuracy"
        );
    }

    #[test]
    fn prompt_names_the_retraction_case_concretely() {
        // The abstract rule alone did not land; naming an instance did.
        let p = build_verify_prompt();
        assert!(p.contains("not John, send it to Jane"));
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
