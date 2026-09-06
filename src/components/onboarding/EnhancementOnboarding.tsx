import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, Sparkles } from "lucide-react";
import { commands } from "@/bindings";
import type { EnhanceModelInfo } from "@/bindings";
import { useEnhanceStore } from "@/stores/enhanceStore";
import { isLocalModelId } from "@/lib/utils/enhanceModels";
import { Button } from "@/components/ui/Button";
import {
  EnhanceModelCard,
  type EnhanceCardStatus,
} from "@/components/settings/models/EnhanceModelCard";
import HandyTextLogo from "../icons/HandyTextLogo";

interface EnhancementOnboardingProps {
  onDone: () => void;
}

/**
 * Offer the enhancement layer during first run.
 *
 * Deliberately not a silent default. The layer rewrites what the user said, and
 * a feature that edits your words should be something you agreed to, not
 * something you discover. So this step asks, and taking it is never required:
 * skipping leaves `enhance_enabled` false, which is also its default, so
 * nothing downloads and nothing changes.
 *
 * Choosing a model does *not* wait for the download. It is a few hundred
 * megabytes, and holding a new user at a progress bar before they have ever
 * dictated anything is the wrong trade — the pipeline already falls back to the
 * raw transcript whenever the model is not ready, so an incomplete download
 * costs nothing but the feature not yet being active.
 */
export const EnhancementOnboarding: React.FC<EnhancementOnboardingProps> = ({
  onDone,
}) => {
  const { t } = useTranslation();
  const {
    models,
    status,
    downloadProgress,
    downloadStats,
    initialize,
    downloadModel,
    selectModel,
  } = useEnhanceStore();
  const [chosenId, setChosenId] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  // A model the user picked off disk cannot exist yet on first run, and the
  // catalogue's own default is the recommendation this step is built around.
  const { recommended, others } = useMemo(() => {
    const catalogue = models.filter((m) => !isLocalModelId(m.id));
    return {
      recommended: catalogue.find((m) => m.default_editor) ?? null,
      others: catalogue.filter((m) => !m.default_editor),
    };
  }, [models]);

  // The sidecar failing to start is not a reason to strand a new user on a step
  // they cannot complete, so treat it as "nothing to offer" and move on.
  useEffect(() => {
    if (status && !status.available) onDone();
  }, [status, onDone]);

  const choose = async (modelId: string) => {
    setBusy(true);
    try {
      await selectModel(modelId);
      setChosenId(modelId);
      // Fire-and-forget: the backend keeps downloading across the transition
      // into the app, and the Models page shows the same progress there.
      void downloadModel(modelId);
    } finally {
      setBusy(false);
    }
  };

  const enableAndFinish = async () => {
    setBusy(true);
    const result = await commands.enhanceSetEnabled(true);
    setBusy(false);
    // A failure here should not trap the user in onboarding; the toggle on the
    // Advanced page is the second chance, and it reports its own errors.
    if (result.status === "error") {
      console.warn("could not enable the enhancement layer:", result.error);
    }
    onDone();
  };

  const cardStatus = (model: EnhanceModelInfo): EnhanceCardStatus => {
    if (downloadProgress[model.id]) return "downloading";
    if (model.downloaded) return model.id === chosenId ? "active" : "available";
    return "downloadable";
  };

  const render = (model: EnhanceModelInfo) => (
    <EnhanceModelCard
      key={model.id}
      model={model}
      status={cardStatus(model)}
      onSelect={(id) => void choose(id)}
      onDownload={(id) => void choose(id)}
      // Nothing is deletable during onboarding: the card only offers it for a
      // model already on disk and selected, which this step never reaches.
      onDelete={() => {}}
      downloadProgress={downloadProgress[model.id]?.percentage}
      downloadSpeed={downloadStats[model.id]?.speed}
    />
  );

  return (
    <div className="h-screen w-screen flex flex-col p-6 gap-4 inset-0">
      <div className="flex flex-col items-center gap-2 shrink-0">
        <HandyTextLogo width={200} />
        <p className="text-text/70 max-w-md font-medium mx-auto">
          {t("onboarding.enhance.subtitle")}
        </p>
      </div>

      <div className="max-w-[600px] w-full mx-auto flex-1 flex flex-col min-h-0 overflow-y-auto">
        <div className="space-y-4 pb-4">
          <div className="flex gap-2 rounded-lg border border-logo-primary/30 bg-logo-primary/5 px-3 py-2 text-left">
            <Sparkles className="w-4 h-4 shrink-0 mt-0.5 text-logo-primary" />
            <p className="text-xs text-text/70 leading-relaxed">
              {t("onboarding.enhance.explainer")}
            </p>
          </div>

          {recommended && render(recommended)}

          {others.length > 0 && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="flex items-center justify-center gap-1.5 mx-auto py-1 text-sm font-medium text-text/60 hover:text-text transition-colors"
            >
              {showAll
                ? t("onboarding.showFewerModels")
                : t("onboarding.enhance.showOthers", { total: others.length })}
              <ChevronDown
                className={`w-4 h-4 transition-transform duration-200 ${
                  showAll ? "rotate-180" : ""
                }`}
              />
            </button>
          )}

          {showAll && others.map(render)}
        </div>
      </div>

      <div className="max-w-[600px] w-full mx-auto shrink-0 flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onDone} disabled={busy}>
          {t("onboarding.enhance.skip")}
        </Button>
        <Button
          variant="primary"
          onClick={() => void enableAndFinish()}
          disabled={!chosenId || busy}
        >
          {t("onboarding.enhance.continue")}
        </Button>
      </div>
    </div>
  );
};

export default EnhancementOnboarding;
