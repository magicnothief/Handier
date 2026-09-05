/**
 * Word-level diff between a raw transcript and its edited version.
 *
 * The enhancement preview exists so people can watch the model act on their own
 * wording before trusting it with dictation, and "it deleted the half-sentence I
 * took back" is not something you can see by reading two paragraphs side by
 * side. Marking the cuts is the whole point of the panel.
 *
 * Matching is done on a normalised key — lowercased, outer punctuation stripped
 * — rather than on the raw token. Without that, repunctuating a sentence makes
 * every word look replaced, and the one edit the user actually needs to check
 * disappears into the noise. A pair that matches on the key but differs in the
 * raw text is reported as `equal` with `tweaked` set, so the UI can show
 * capitalisation and punctuation fixes without shouting about them.
 */

/** One span of the diff, in the order it should be rendered. */
export interface DiffOp {
  type: "equal" | "insert" | "delete";
  /** The word as written, without its trailing whitespace. */
  text: string;
  /** Whitespace that followed the word in its source text. */
  space: string;
  /**
   * For `equal`: how the same word was spelled in the original.
   *
   * The two rows show different sides of the pair, so a matched word cannot be
   * carried as one string: the "you said" row has to keep the speaker's own
   * capitalisation even where the edit changed it.
   */
  from?: { text: string; space: string };
  /** For `equal`: the two sides differ only in case or punctuation. */
  tweaked?: boolean;
}

interface Token {
  raw: string;
  key: string;
  space: string;
}

/** Longest input the quadratic matcher will attempt, in words per side. */
const MAX_TOKENS = 600;

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  for (const match of text.matchAll(/(\S+)(\s*)/g)) {
    const raw = match[1];
    tokens.push({
      raw,
      // Strip only the outer punctuation: an internal apostrophe or hyphen is
      // part of the word ("don't", "twenty-five") and dropping it would match
      // words the speaker did not say.
      key: raw.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""),
      space: match[2],
    });
  }
  return tokens;
}

/**
 * Diff `original` against `edited`, word by word.
 *
 * Returns a single ordered list: render `equal` and `delete` for the "before"
 * side, `equal` and `insert` for the "after" side.
 */
export function diffWords(original: string, edited: string): DiffOp[] {
  const a = tokenize(original);
  const b = tokenize(edited);

  // A transcript this long is not a dictated sentence, and the table below is
  // quadratic. Fall back to "everything was replaced", which still renders.
  if (a.length > MAX_TOKENS || b.length > MAX_TOKENS) {
    return [
      ...a.map(
        (t): DiffOp => ({ type: "delete", text: t.raw, space: t.space }),
      ),
      ...b.map(
        (t): DiffOp => ({ type: "insert", text: t.raw, space: t.space }),
      ),
    ];
  }

  // Classic LCS table. `lcs[i][j]` is the length of the longest common
  // subsequence of `a[i..]` and `b[j..]`, filled from the end so the walk below
  // can emit spans in reading order.
  const lcs: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0),
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] =
        a[i].key === b[j].key
          ? lcs[i + 1][j + 1] + 1
          : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i].key === b[j].key) {
      ops.push({
        type: "equal",
        // `text` is the edited spelling — the one that will be pasted.
        text: b[j].raw,
        space: b[j].space,
        from: { text: a[i].raw, space: a[i].space },
        tweaked: a[i].raw !== b[j].raw,
      });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      ops.push({ type: "delete", text: a[i].raw, space: a[i].space });
      i++;
    } else {
      ops.push({ type: "insert", text: b[j].raw, space: b[j].space });
      j++;
    }
  }
  for (; i < a.length; i++) {
    ops.push({ type: "delete", text: a[i].raw, space: a[i].space });
  }
  for (; j < b.length; j++) {
    ops.push({ type: "insert", text: b[j].raw, space: b[j].space });
  }
  return ops;
}

/** Whether a diff contains anything worth highlighting. */
export function hasVisibleChange(ops: DiffOp[]): boolean {
  return ops.some((op) => op.type !== "equal" || op.tweaked);
}

/** The spelling and spacing to render for `op` on the given side. */
export function sideOf(
  op: DiffOp,
  side: "before" | "after",
): { text: string; space: string } {
  if (side === "before" && op.from) return op.from;
  return { text: op.text, space: op.space };
}
