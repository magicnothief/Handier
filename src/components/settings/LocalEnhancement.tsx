import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands } from "../../bindings";
import type { EnhanceModelInfo, EnhanceStatus } from "../../bindings";
import { useSettings } from "../../hooks/useSettings";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";
import { SettingContainer } from "../ui/SettingContainer";
import { SettingsGroup } from "../ui/SettingsGroup";
import { Textarea } from "../ui/Textarea";
import { ToggleSwitch } from "../ui/ToggleSwitch";

type PreviewResult = {
  text: string;
  original: string;
  changed: boolean;
  rejected: string | null;
  verdict: string | null;
  error: string | null;
  elapsedMs: number;
};

/**
 * Settings for the on-device transcript enhancement layer.
 *
 * Everything here is local: the model selector downloads a GGUF and the rest
 * configures how aggressively it may rewrite what was said. The preview exists
 * because "cut what I retracted" is hard to judge in the abstract — people need
 * to see it act on their own wording before trusting it with dictation.
 */
export const LocalEnhancement: React.FC = React.memo(() => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const [status, setStatus] = useState<EnhanceStatus | null>(null);
  const [models, setModels] = useState<EnhanceModelInfo[]>([]);
  const [busyModelId, setBusyModelId] = useState<string | null>(null);
  const [previewInput, setPreviewInput] = useState("");
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enabled = getSetting("enhance_enabled") ?? false;
  const options = getSetting("enhance_options");
  const selectedModelId = getSetting("enhance_model_id") ?? null;

  const refresh = useCallback(async () => {
    const [statusResult, modelsResult] = await Promise.all([
      commands.enhanceStatus(),
      commands.enhanceListModels("editor"),
    ]);
    if (statusResult.status === "ok") setStatus(statusResult.data);
    if (modelsResult.status === "ok") setModels(modelsResult.data);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const activeModel = useMemo(
    () =>
      models.find((m) => m.id === selectedModelId) ??
      models.find((m) => m.default_editor),
    [models, selectedModelId],
  );

  const modelOptions = useMemo(
    () =>
      models.map((m) => ({
        value: m.id,
        label: `${m.name} · ${t("settings.localEnhancement.model.size", {
          size: Math.round(m.size_bytes / 1048576),
        })} · ${m.downloaded ? t("settings.localEnhancement.model.downloaded") : t("settings.localEnhancement.model.notDownloaded")}`,
      })),
    [models, t],
  );

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

  const download = useCallback(
    async (modelId: string) => {
      setBusyModelId(modelId);
      setError(null);
      const result = await commands.enhanceDownloadModel(modelId);
      if (result.status === "error") setError(result.error);
      setBusyModelId(null);
      void refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (modelId: string) => {
      setBusyModelId(modelId);
      const result = await commands.enhanceDeleteModel(modelId);
      if (result.status === "error") setError(result.error);
      setBusyModelId(null);
      void refresh();
    },
    [refresh],
  );

  const runPreview = useCallback(async () => {
    setPreviewing(true);
    setError(null);
    setPreview(null);
    const result = await commands.enhancePreview(previewInput);
    if (result.status === "ok") {
      setPreview(result.data as PreviewResult);
    } else {
      setError(result.error);
    }
    setPreviewing(false);
    void refresh();
  }, [previewInput, refresh]);

  const toggleEnabled = useCallback(
    async (next: boolean) => {
      setError(null);
      const result = await commands.enhanceSetEnabled(next);
      if (result.status === "error") setError(result.error);
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
        isUpdating={isUpdating("enhance_enabled")}
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
        <div className="flex items-center gap-2">
          <Select
            value={activeModel?.id ?? null}
            options={modelOptions}
            placeholder={t("settings.localEnhancement.model.placeholder")}
            onChange={(value) => void updateSetting("enhance_model_id", value)}
            className="min-w-[18rem]"
          />
          {activeModel && !activeModel.downloaded && (
            <Button
              variant="primary"
              onClick={() => void download(activeModel.id)}
              disabled={busyModelId === activeModel.id}
            >
              {busyModelId === activeModel.id
                ? t("settings.localEnhancement.model.downloading")
                : t("settings.localEnhancement.model.download")}
            </Button>
          )}
          {activeModel?.downloaded && (
            <Button
              variant="secondary"
              onClick={() => void remove(activeModel.id)}
              disabled={busyModelId === activeModel.id}
            >
              {t("settings.localEnhancement.model.delete")}
            </Button>
          )}
        </div>
        {activeModel && (
          <p className="text-xs text-mid-gray/70 mt-1">
            {activeModel.description}{" "}
            {t("settings.localEnhancement.model.ram", {
              ram: activeModel.min_ram_mb,
            })}{" "}
            ·{" "}
            {t("settings.localEnhancement.model.released", {
              date: activeModel.released,
            })}
          </p>
        )}
      </SettingContainer>

      {options && (
        <>
          <ToggleSwitch
            checked={options.removeFillers}
            onChange={(v) => setOption("removeFillers", v)}
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
          >
            <Select
              value={options.verify}
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

      <SettingContainer
        title={t("settings.localEnhancement.preview.title")}
        description={t("settings.localEnhancement.preview.description")}
        descriptionMode="inline"
        grouped
      >
        <div className="flex flex-col gap-2 w-full">
          <Textarea
            value={previewInput}
            onChange={(e) => setPreviewInput(e.target.value)}
            placeholder={t("settings.localEnhancement.preview.placeholder")}
            rows={2}
          />
          <div>
            <Button
              variant="primary"
              onClick={() => void runPreview()}
              disabled={previewing || previewInput.trim().length === 0}
            >
              {previewing
                ? t("settings.localEnhancement.preview.running")
                : t("settings.localEnhancement.preview.run")}
            </Button>
          </div>
          {preview && (
            <div className="text-sm border border-mid-gray/20 rounded-md p-2">
              <p className="whitespace-pre-wrap">{preview.text}</p>
              <p className="text-xs text-mid-gray/70 mt-1">
                {!preview.changed &&
                  t("settings.localEnhancement.preview.unchanged")}{" "}
                {preview.rejected &&
                  t("settings.localEnhancement.preview.rejected", {
                    reason: preview.rejected,
                  })}{" "}
                {preview.verdict === "Changed" &&
                  t("settings.localEnhancement.preview.verdictChanged")}{" "}
                {t("settings.localEnhancement.preview.took", {
                  ms: preview.elapsedMs,
                })}
              </p>
            </div>
          )}
        </div>
      </SettingContainer>

      {error && <p className="text-sm text-red-400 px-2 py-1">{error}</p>}
    </SettingsGroup>
  );
});
