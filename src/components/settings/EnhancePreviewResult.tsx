import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ArrowDown } from "lucide-react";
import type { EnhancePreview } from "@/bindings";
import { diffWords, hasVisibleChange, sideOf } from "@/lib/utils/wordDiff";

interface EnhancePreviewResultProps {
  preview: EnhancePreview;
}

/**
 * The result of a preview run, shown as a marked-up before/after pair.
 *
 * Plain output text was not enough to judge this feature by. The thing people
 * need to check is whether the model cut the half-sentence they took back and
 * nothing else, and comparing two paragraphs by eye does not answer that — the
 * one deletion that matters looks identical to the twenty words that did not
 * move. Marking the cuts turns "read both and hope" into a glance.
 */
export const EnhancePreviewResult: React.FC<EnhancePreviewResultProps> = ({
  preview,
}) => {
  const { t } = useTranslation();
  const ops = useMemo(
    () => diffWords(preview.original, preview.text),
    [preview.original, preview.text],
  );
  const changed = hasVisibleChange(ops);
  const cut = ops.some((op) => op.type === "delete");
  const added = ops.some((op) => op.type === "insert");

  return (
    <div className="rounded-lg border border-mid-gray/20 overflow-hidden">
      <DiffRow
        label={t("settings.localEnhancement.preview.before")}
        ops={ops}
        side="before"
      />
      <div className="flex items-center gap-2 px-3 py-1 bg-mid-gray/5 border-y border-mid-gray/20">
        <ArrowDown className="w-3 h-3 text-text/40" />
        <span className="text-[11px] uppercase tracking-wide text-text/40">
          {t("settings.localEnhancement.preview.after")}
        </span>
      </div>
      <DiffRow ops={ops} side="after" />

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 border-t border-mid-gray/20 text-xs text-text/50">
        <span className="tabular-nums">
          {t("settings.localEnhancement.preview.took", {
            ms: preview.elapsedMs,
          })}
        </span>
        {!changed && (
          <span>{t("settings.localEnhancement.preview.unchanged")}</span>
        )}
        {preview.rejected && (
          <span className="text-amber-600 dark:text-amber-400">
            {t("settings.localEnhancement.preview.rejected", {
              reason: preview.rejected,
            })}
          </span>
        )}
        {preview.verdict === "Changed" && (
          <span className="text-amber-600 dark:text-amber-400">
            {t("settings.localEnhancement.preview.verdictChanged")}
          </span>
        )}
        {changed && <Legend cut={cut} added={added} />}
      </div>
    </div>
  );
};

interface DiffRowProps {
  label?: string;
  ops: ReturnType<typeof diffWords>;
  side: "before" | "after";
}

/**
 * One side of the pair.
 *
 * Only the side that owns a span is marked: a cut is called out on the "you
 * said" row where it was removed from, an addition on the row it appears in.
 * Marking both sides doubles the ink for the same fact.
 *
 * Exported for History, which shows the "you said" side under each edited
 * entry. One rendering of "what was cut" keeps the preview and the record of
 * real dictations reading the same way.
 */
export const DiffRow: React.FC<DiffRowProps> = ({ label, ops, side }) => {
  const hidden = side === "before" ? "insert" : "delete";
  return (
    <div className="px-3 py-2">
      {label && (
        <div className="text-[11px] uppercase tracking-wide text-text/40 mb-1">
          {label}
        </div>
      )}
      <p className="text-sm whitespace-pre-wrap leading-relaxed">
        {ops.map((op, index) => {
          if (op.type === hidden) return null;
          const { text, space } = sideOf(op, side);
          let className = "";
          if (op.type === "delete") {
            className =
              "line-through decoration-2 text-red-600/70 dark:text-red-400/70 bg-red-500/10 rounded-sm px-0.5";
          } else if (op.type === "insert") {
            className =
              "text-green-700 dark:text-green-400 bg-green-500/10 rounded-sm px-0.5";
          } else if (op.tweaked && side === "after") {
            // Capitalisation and punctuation fixes are real edits, but they are
            // not the ones a user is checking for. A dotted underline says
            // "this moved" without competing with a deleted clause.
            className =
              "underline decoration-dotted decoration-text/50 underline-offset-[3px]";
          }
          return (
            <React.Fragment key={index}>
              <span className={className}>{text}</span>
              {space || " "}
            </React.Fragment>
          );
        })}
      </p>
    </div>
  );
};

/**
 * What the colours mean, spelled out rather than left to be inferred.
 *
 * Only the keys actually used are listed. A legend entry for a colour that does
 * not appear sends the reader hunting for green words in a diff that only cut
 * things.
 */
const Legend: React.FC<{ cut: boolean; added: boolean }> = ({ cut, added }) => {
  const { t } = useTranslation();
  if (!cut && !added) return null;
  return (
    <span className="flex items-center gap-3 ms-auto">
      {cut && (
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-red-500/40" />
          {t("settings.localEnhancement.preview.legendCut")}
        </span>
      )}
      {added && (
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-green-500/40" />
          {t("settings.localEnhancement.preview.legendAdded")}
        </span>
      )}
    </span>
  );
};

export default EnhancePreviewResult;
