import React, { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ask, open } from "@tauri-apps/plugin-dialog";
import { FolderOpen } from "lucide-react";
import { useEnhanceStore } from "@/stores/enhanceStore";
import { useSettings } from "@/hooks/useSettings";
import type { EnhanceModelInfo, PromptStyle } from "@/bindings";
import { isLocalModelId, localModelId } from "@/lib/utils/enhanceModels";
import { Button } from "@/components/ui/Button";
import { EnhanceModelCard, type EnhanceCardStatus } from "./EnhanceModelCard";

/**
 * Enhancement models, listed alongside the transcription models.
 *
 * They live on this page rather than buried in advanced settings because they
 * are the same kind of thing to a user: a model you download, pick, and delete.
 * Splitting them across two places meant the enhancement layer read as a bolted-on
 * extra rather than part of the app.
 */
export const EnhanceModelsSection: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting } = useSettings();
  const {
    models,
    status,
    downloadProgress,
    downloadStats,
    loadingModelId,
    error,
    initialize,
    downloadModel,
    deleteModel,
    selectModel,
    setError,
  } = useEnhanceStore();

  useEffect(() => {
    void initialize();
  }, [initialize]);

  const selectedModelId = getSetting("enhance_model_id") ?? null;
  const promptStyle = getSetting("enhance_prompt_style") ?? null;

  // With no explicit choice the backend falls back to the catalog default, so
  // the card marked active has to follow the same rule or the page disagrees
  // with what actually runs.
  const activeId = useMemo(() => {
    if (selectedModelId && models.some((m) => m.id === selectedModelId)) {
      return selectedModelId;
    }
    return models.find((m) => m.default_editor)?.id ?? null;
  }, [models, selectedModelId]);

  const getStatus = (model: EnhanceModelInfo): EnhanceCardStatus => {
    if (downloadProgress[model.id]) return "downloading";
    if (loadingModelId === model.id) return "loading";
    if (!model.downloaded) return "downloadable";
    return model.id === activeId ? "active" : "available";
  };

  const handleDelete = async (modelId: string) => {
    // A model loaded off disk is the user's file. "Removing" it means going
    // back to the catalog default, never touching what is on their disk.
    if (isLocalModelId(modelId)) {
      await selectModel(null);
      return;
    }
    const model = models.find((m) => m.id === modelId);
    const modelName = model?.name ?? modelId;
    const confirmed = await ask(
      modelId === activeId
        ? t("settings.models.deleteActiveConfirm", { modelName })
        : t("settings.models.deleteConfirm", { modelName }),
      { title: t("settings.models.deleteTitle"), kind: "warning" },
    );
    if (confirmed) await deleteModel(modelId);
  };

  const handlePickLocal = async () => {
    setError(null);
    const picked = await open({
      multiple: false,
      directory: false,
      title: t("settings.models.enhance.local.pickTitle"),
      filters: [
        {
          name: t("settings.models.enhance.local.fileType"),
          extensions: ["gguf"],
        },
      ],
    });
    if (typeof picked !== "string") return;
    await selectModel(localModelId(picked));
  };

  const { downloaded, available, local } = useMemo(() => {
    const has: EnhanceModelInfo[] = [];
    const rest: EnhanceModelInfo[] = [];
    let own: EnhanceModelInfo | null = null;
    for (const model of models) {
      if (isLocalModelId(model.id)) own = model;
      else if (model.downloaded || downloadProgress[model.id]) has.push(model);
      else rest.push(model);
    }
    // Active first, then the recommended default, so the two cards a user most
    // likely wants are at the top of each list.
    const rank = (m: EnhanceModelInfo) =>
      m.id === activeId ? 0 : m.default_editor ? 1 : 2;
    has.sort((a, b) => rank(a) - rank(b));
    rest.sort((a, b) => rank(a) - rank(b));
    return { downloaded: has, available: rest, local: own };
  }, [models, downloadProgress, activeId]);

  if (status && !status.available) {
    return (
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">
          {t("settings.models.enhance.title")}
        </h2>
        <p className="text-sm text-text/50">
          {status.error ?? t("settings.localEnhancement.unavailable")}
        </p>
      </div>
    );
  }

  const render = (model: EnhanceModelInfo) => (
    <EnhanceModelCard
      key={model.id}
      model={model}
      status={getStatus(model)}
      onSelect={(id) => void selectModel(id)}
      onDownload={(id) => void downloadModel(id)}
      onDelete={(id) => void handleDelete(id)}
      downloadProgress={downloadProgress[model.id]?.percentage}
      downloadSpeed={downloadStats[model.id]?.speed}
    />
  );

  return (
    <div className="space-y-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">
          {t("settings.models.enhance.title")}
        </h2>
        <p className="text-sm text-text/60">
          {t("settings.models.enhance.description")}
        </p>
      </div>

      <div className="space-y-3">
        <h3 className="text-sm font-medium text-text/60">
          {t("settings.models.yourModels")}
        </h3>
        {downloaded.length > 0 ? (
          downloaded.map(render)
        ) : (
          <p className="text-sm text-text/50">
            {t("settings.models.enhance.noneDownloaded")}
          </p>
        )}
      </div>

      {available.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-text/60">
            {t("settings.models.enhance.available")}
          </h3>
          {available.map(render)}
        </div>
      )}

      <div className="space-y-3">
        <div>
          <h3 className="text-sm font-medium text-text/60">
            {t("settings.models.enhance.local.title")}
          </h3>
          <p className="text-xs text-text/50 mt-1">
            {t("settings.models.enhance.local.description")}
          </p>
        </div>
        {local && render(local)}
        {local && (
          <PromptStyleChoice
            value={promptStyle}
            onChange={(next) =>
              void updateSetting("enhance_prompt_style", next)
            }
          />
        )}
        <Button
          variant="secondary"
          size="sm"
          onClick={() => void handlePickLocal()}
          className="flex items-center gap-1.5"
        >
          <FolderOpen className="w-3.5 h-3.5" />
          <span>
            {local
              ? t("settings.models.enhance.local.replace")
              : t("settings.models.enhance.local.pick")}
          </span>
        </Button>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
};

interface PromptStyleChoiceProps {
  value: PromptStyle | null;
  onChange: (value: PromptStyle | null) => void;
}

/**
 * How the picked file should be prompted.
 *
 * Exposed because the app cannot tell a fine-tune from a stock model by looking
 * at a GGUF, and getting it wrong is not subtle: a stock model with no prompt
 * does not edit at all, and a fine-tune handed the instruction prompt starts
 * copying the prompt's own rules into the user's text. Defaults to "tuned",
 * which is why someone would point this at their own weights in the first place.
 */
const PromptStyleChoice: React.FC<PromptStyleChoiceProps> = ({
  value,
  onChange,
}) => {
  const { t } = useTranslation();
  const options: { key: PromptStyle; label: string; hint: string }[] = [
    {
      key: "tuned",
      label: t("settings.models.enhance.local.styleTuned"),
      hint: t("settings.models.enhance.local.styleTunedHint"),
    },
    {
      key: "alpaca",
      label: t("settings.models.enhance.local.styleAlpaca"),
      hint: t("settings.models.enhance.local.styleAlpacaHint"),
    },
    {
      key: "instructed",
      label: t("settings.models.enhance.local.styleInstructed"),
      hint: t("settings.models.enhance.local.styleInstructedHint"),
    },
  ];
  const current = value ?? "tuned";

  return (
    <div className="rounded-xl border-2 border-mid-gray/20 px-4 py-3 space-y-2">
      <p className="text-sm font-medium text-text/80">
        {t("settings.models.enhance.local.styleTitle")}
      </p>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option.key}
            type="button"
            onClick={() => onChange(option.key)}
            title={option.hint}
            className={`rounded-lg px-3 py-1.5 text-xs transition-colors ${
              current === option.key
                ? "bg-logo-primary/15 text-logo-primary border border-logo-primary/40"
                : "border border-mid-gray/20 text-text/60 hover:text-text hover:border-logo-primary/40"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-text/50">
        {options.find((o) => o.key === current)?.hint}
      </p>
    </div>
  );
};

export default EnhanceModelsSection;
