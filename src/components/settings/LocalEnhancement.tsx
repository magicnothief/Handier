import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Sparkles } from "lucide-react";
import { commands } from "../../bindings";
import type { EnhancePreview } from "../../bindings";
import { useSettings } from "../../hooks/useSettings";
import { useEnhanceStore } from "../../stores/enhanceStore";
import { useNavStore } from "../../stores/navStore";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { Textarea } from "../ui/Textarea";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { EnhancePreviewResult } from "./EnhancePreviewResult";

/**
 * Settings for the on-device transcript enhancement layer.
 *
 * Everything here is local: this page configures how aggressively the model may
 * rewrite what was said, and the preview exists because "cut what I retracted"
 * is hard to judge in the abstract — people need to see it act on their own
 * wording before trusting it with dictation.
 *
 * Downloading and picking a model deliberately does *not* happen here. It
 * happens on the Models page next to the transcription models, because to a
 * user those are the same kind of thing. What is left here is a read-only
 * summary of which model is in play and a way to get to that page.
 */
export const LocalEnhancement: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const goToSection = useNavStore((state) => state.setSection);
  const { models, status, downloadProgress, initialize, refresh } =
    useEnhanceStore();

  const [previewInput, setPreviewInput] = useState("");
  const [preview, setPreview] = useState<EnhancePreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [togglingEnabled, setTogglingEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enabled = getSetting("enhance_enabled") ?? false;
  const options = getSetting("enhance_options");
  const selectedModelId = getSetting("enhance_model_id") ?? null;

  useEffect(() => {
    void initialize();
  }, [initialize]);

  // No explicit choice means the backend runs the catalog default, so the model
  // named here has to follow the same rule or this page describes a model that
  // is not the one doing the work.
  const activeModel = useMemo(
    () =>
      models.find((m) => m.id === selectedModelId) ??
      models.find((m) => m.default_editor),
    [models, selectedModelId],
  );

  const activeProgress = activeModel ? downloadProgress[activeModel.id] : null;

  // Only a stock model is sent the prompt these switches assemble. A fine-tune
  // gets an empty system turn or one fixed Alpaca instruction, so every switch
  // below has no effect on it. Leaving them live would make the page lie: the
  // user would flip "remove filler words", see it stick, and get identical
  // output. Written as a positive test against `instructed` so a prompt style
  // added later is treated as fixed until someone says otherwise.
  // `?? "instructed"` so the switches stay live while the model list is still
  // loading, rather than flashing the notice with an empty model name.
  const promptShapingApplies =
    (activeModel?.prompt_style ?? "instructed") === "instructed";

  const modelState = useMemo(() => {
    if (!activeModel) return t("settings.localEnhancement.model.none");
    if (activeProgress) {
      return t("settings.localEnhancement.model.downloadingPercent", {
        percent: Math.floor(activeProgress.percentage),
      });
    }
    return activeModel.downloaded
      ? t("settings.localEnhancement.model.ready")
      : t("settings.localEnhancement.model.notDownloaded");
  }, [activeModel, activeProgress, t]);

  const setOption = useCallback(
    <K extends keyof NonNullable<typeof options>>(
      key: K,
      value: NonNullable<typeof options>[K],
    ) => {
      if (!options) return;
      void updateSetting("enhance_options", { ...options, [key]: value });
    },
    [options, updateSetting],
  );

  const runPreview = useCallback(async () => {
    setPreviewing(true);
    setError(null);
    setPreview(null);
    const result = await commands.enhancePreview(previewInput);
    if (result.status === "ok") {
      setPreview(result.data);
    } else {
      setError(result.error);
    }
    setPreviewing(false);
    void refresh();
  }, [previewInput, refresh]);

  // Turning the layer on can preload the model, which takes seconds. Without a
  // pending state the switch sits inert for that whole time and reads as
  // broken, so hold it busy until the command returns. The switch's own
  // position comes from the settings store, which the backend now refreshes by
  // emitting `settings-changed` -- `isUpdating` never covered this path,
  // because that flag belongs to `updateSetting` and this is a command.
  const toggleEnabled = useCallback(
    async (next: boolean) => {
      setError(null);
      setTogglingEnabled(true);
      try {
        const result = await commands.enhanceSetEnabled(next);
        if (result.status === "error") setError(result.error);
      } finally {
        setTogglingEnabled(false);
      }
      void refresh();
    },
    [refresh],
  );

  if (status && !status.available) {
    return (
      <SettingsGroup title={t("settings.localEnhancement.title")}>
        <p className="text-sm text-mid-gray/80 px-2 py-1">
          {status.error ?? t("settings.localEnhancement.unavailable")}
        </p>
      </SettingsGroup>
    );
  }

  return (
    <SettingsGroup title={t("settings.localEnhancement.title")}>
      <p className="text-sm text-mid-gray/80 px-2 pb-2">
        {t("settings.localEnhancement.description")}
      </p>

      <ToggleSwitch
        checked={enabled}
        onChange={toggleEnabled}
        isUpdating={togglingEnabled || isUpdating("enhance_enabled")}
        label={t("settings.localEnhancement.enable.title")}
        description={t("settings.localEnhancement.enable.description")}
        descriptionMode="inline"
        grouped
      />

      <SettingContainer
        title={t("settings.localEnhancement.model.title")}
        description={t("settings.localEnhancement.model.description")}
        descriptionMode="inline"
        grouped
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex flex-col items-end text-right min-w-0">
            {/* Bounded, not just `min-w-0`: a truncating span with no maximum
                still grows to its content, and a model the user named himself
                can be long enough to run back over the description column. */}
            <span
              className="text-sm text-text truncate max-w-[9rem]"
              title={activeModel?.name ?? undefined}
            >
              {activeModel?.name ?? t("settings.localEnhancement.model.none")}
            </span>
            <span className="text-xs text-mid-gray/70">{modelState}</span>
          </div>
          <Button
            variant="secondary"
            onClick={() => goToSection("models", "enhancement-models")}
            className="shrink-0 whitespace-nowrap"
          >
            {t("settings.localEnhancement.model.manage")}
          </Button>
        </div>
      </SettingContainer>

      {options && !promptShapingApplies && (
        <div className="mx-2 mb-2 flex gap-2 rounded-lg border border-logo-primary/30 bg-logo-primary/5 px-3 py-2">
          <Sparkles className="w-4 h-4 shrink-0 mt-0.5 text-logo-primary" />
          <p className="text-xs text-text/70 leading-relaxed">
            {t("settings.localEnhancement.tunedNotice", {
              modelName: activeModel?.name ?? "",
            })}
          </p>
        </div>
      )}

      {options && (
        <>
          <ToggleSwitch
            checked={options.removeFillers}
            onChange={(v) => setOption("removeFillers", v)}
            disabled={!promptShapingApplies}
            label={t("settings.localEnhancement.features.removeFillers.title")}
            description={t(
              "settings.localEnhancement.features.removeFillers.description",
            )}
            descriptionMode="inline"
            grouped
          />
          <ToggleSwitch
            checked={options.fixSelfCorrections}
            onChange={(v) => setOption("fixSelfCorrections", v)}
            disabled={!promptShapingApplies}
            label={t(
              "settings.localEnhancement.features.fixSelfCorrections.title",
            )}
            description={t(
              "settings.localEnhancement.features.fixSelfCorrections.description",
            )}
            descriptionMode="inline"
            grouped
          />
          <ToggleSwitch
            checked={options.fixPunctuation}
            onChange={(v) => setOption("fixPunctuation", v)}
            disabled={!promptShapingApplies}
            label={t("settings.localEnhancement.features.fixPunctuation.title")}
            description={t(
              "settings.localEnhancement.features.fixPunctuation.description",
            )}
            descriptionMode="inline"
            grouped
          />
          <ToggleSwitch
            checked={options.formatStructure}
            onChange={(v) => setOption("formatStructure", v)}
            disabled={!promptShapingApplies}
            label={t(
              "settings.localEnhancement.features.formatStructure.title",
            )}
            description={t(
              "settings.localEnhancement.features.formatStructure.description",
            )}
            descriptionMode="inline"
            grouped
          />
          <ToggleSwitch
            checked={options.spokenCommands}
            onChange={(v) => setOption("spokenCommands", v)}
            disabled={!promptShapingApplies}
            label={t("settings.localEnhancement.features.spokenCommands.title")}
            description={t(
              "settings.localEnhancement.features.spokenCommands.description",
            )}
            descriptionMode="inline"
            grouped
          />
          <ToggleSwitch
            checked={options.contextAwareTone}
            onChange={(v) => setOption("contextAwareTone", v)}
            disabled={!promptShapingApplies}
            label={t(
              "settings.localEnhancement.features.contextAwareTone.title",
            )}
            description={t(
              "settings.localEnhancement.features.contextAwareTone.description",
            )}
            descriptionMode="inline"
            grouped
          />

          <SettingContainer
            title={t("settings.localEnhancement.aggressiveness.title")}
            description={t(
              "settings.localEnhancement.aggressiveness.description",
            )}
            descriptionMode="inline"
            grouped
          >
            <Select
              value={options.aggressiveness}
              options={[
                {
                  value: "light",
                  label: t("settings.localEnhancement.aggressiveness.light"),
                },
                {
                  value: "balanced",
                  label: t("settings.localEnhancement.aggressiveness.balanced"),
                },
                {
                  value: "aggressive",
                  label: t(
                    "settings.localEnhancement.aggressiveness.aggressive",
                  ),
                },
              ]}
              onChange={(v) =>
                v &&
                setOption("aggressiveness", v as typeof options.aggressiveness)
              }
              className="min-w-[18rem]"
            />
          </SettingContainer>

          <SettingContainer
            title={t("settings.localEnhancement.verify.title")}
            description={t("settings.localEnhancement.verify.description")}
            descriptionMode="inline"
            grouped
            disabled={!promptShapingApplies}
          >
            <Select
              disabled={!promptShapingApplies}
              value={promptShapingApplies ? options.verify : "off"}
              options={[
                {
                  value: "off",
                  label: t("settings.localEnhancement.verify.off"),
                },
                {
                  value: "auto",
                  label: t("settings.localEnhancement.verify.auto"),
                },
                {
                  value: "always",
                  label: t("settings.localEnhancement.verify.always"),
                },
              ]}
              onChange={(v) =>
                v && setOption("verify", v as typeof options.verify)
              }
              className="min-w-[18rem]"
            />
          </SettingContainer>
        </>
      )}

      <ToggleSwitch
        checked={getSetting("enhance_use_gpu") ?? true}
        onChange={(v) => void updateSetting("enhance_use_gpu", v)}
        isUpdating={isUpdating("enhance_use_gpu")}
        label={t("settings.localEnhancement.useGpu.title")}
        description={t("settings.localEnhancement.useGpu.description")}
        descriptionMode="inline"
        grouped
      />
      <ToggleSwitch
        checked={getSetting("enhance_keep_loaded") ?? true}
        onChange={(v) => void updateSetting("enhance_keep_loaded", v)}
        isUpdating={isUpdating("enhance_keep_loaded")}
        label={t("settings.localEnhancement.keepLoaded.title")}
        description={t("settings.localEnhancement.keepLoaded.description")}
        descriptionMode="inline"
        grouped
      />

      {/* Stacked: the textarea and the marked-up result are the point of this
          row, and the control column of a horizontal setting is a ~270px
          gutter that wrapped the placeholder over three lines. */}
      <SettingContainer
        title={t("settings.localEnhancement.preview.title")}
        description={t("settings.localEnhancement.preview.description")}
        descriptionMode="inline"
        layout="stacked"
        grouped
      >
        <div className="flex flex-col gap-2 w-full">
          <Textarea
            value={previewInput}
            onChange={(e) => setPreviewInput(e.target.value)}
            placeholder={t("settings.localEnhancement.preview.placeholder")}
            rows={2}
          />
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              onClick={() => void runPreview()}
              disabled={
                previewing ||
                previewInput.trim().length === 0 ||
                !activeModel?.downloaded
              }
            >
              {previewing
                ? t("settings.localEnhancement.preview.running")
                : t("settings.localEnhancement.preview.run")}
            </Button>
            {previewing && (
              // The first preview after launch also pays for loading the
              // model, which is tens of seconds. Unannounced, that reads as a
              // hang rather than as work in progress.
              <span className="text-xs text-mid-gray/70">
                {t("settings.localEnhancement.preview.firstRunHint")}
              </span>
            )}
            {!previewing && !activeModel?.downloaded && (
              <span className="text-xs text-mid-gray/70">
                {t("settings.localEnhancement.preview.needsModel")}
              </span>
            )}
          </div>
          {preview && <EnhancePreviewResult preview={preview} />}
        </div>
      </SettingContainer>

      {error && <p className="text-sm text-red-400 px-2 py-1">{error}</p>}
    </SettingsGroup>
  );
});
