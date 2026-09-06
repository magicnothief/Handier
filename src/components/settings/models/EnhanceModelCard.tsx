import React from "react";
import { useTranslation } from "react-i18next";
import {
  Check,
  Cpu,
  Download,
  HardDrive,
  Loader2,
  MemoryStick,
  Trash2,
  X,
} from "lucide-react";
import type { EnhanceModelInfo } from "@/bindings";
import { isLocalModelId } from "@/lib/utils/enhanceModels";
import { formatModelSize } from "@/lib/utils/format";
import Badge from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

export type EnhanceCardStatus =
  | "downloadable"
  | "downloading"
  | "loading"
  | "active"
  | "available";

interface EnhanceModelCardProps {
  model: EnhanceModelInfo;
  status: EnhanceCardStatus;
  onSelect: (modelId: string) => void;
  onDownload: (modelId: string) => void;
  /**
   * Omit where deleting makes no sense — onboarding, for instance, where the
   * card is a choice rather than a management surface. The control is hidden
   * rather than inert: a button that does nothing is worse than no button.
   */
  onDelete?: (modelId: string) => void;
  downloadProgress?: number;
  downloadSpeed?: number;
}

/**
 * One enhancement model, presented the way transcription models are.
 *
 * A sibling of `ModelCard` rather than a generalisation of it: that component
 * is built around `ModelInfo`'s accuracy scores, language lists and streaming
 * flags, none of which an enhancement model has. Sharing the markup would mean
 * threading a dozen optional props through the page every transcription model
 * already depends on. Sharing the *visual language* is what matters here, and
 * that comes from using the same classes, badges and buttons.
 */
export const EnhanceModelCard: React.FC<EnhanceModelCardProps> = ({
  model,
  status,
  onSelect,
  onDownload,
  onDelete,
  downloadProgress,
  downloadSpeed,
}) => {
  const { t } = useTranslation();
  const isClickable = status === "available" || status === "downloadable";
  // A model the user picked off disk is theirs: the app knows nothing about it
  // beyond the file, and it must never offer to delete it.
  const isLocal = isLocalModelId(model.id);

  const handleClick = () => {
    if (!isClickable) return;
    if (status === "downloadable") onDownload(model.id);
    else onSelect(model.id);
  };

  const variantClasses =
    status === "active"
      ? "border-2 border-logo-primary/50 bg-logo-primary/10"
      : "border-2 border-mid-gray/20";

  const interactiveClasses = isClickable
    ? "cursor-pointer hover:border-logo-primary/50 hover:bg-logo-primary/5 hover:shadow-lg hover:scale-[1.01] active:scale-[0.99] group"
    : "";

  return (
    <div
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" && isClickable) handleClick();
      }}
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      className={`flex flex-col rounded-xl px-4 py-3 gap-2 text-left transition-all duration-200 ${variantClasses} ${interactiveClasses}`}
    >
      <div className="flex justify-between items-center w-full">
        <div className="flex flex-col items-start flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h3
              className={`text-base font-semibold text-text ${isClickable ? "group-hover:text-logo-primary" : ""} transition-colors`}
            >
              {model.name}
            </h3>
            {isLocal && (
              <Badge variant="secondary">
                {t("settings.models.enhance.local.badge")}
              </Badge>
            )}
            {model.default_editor && status !== "active" && (
              <Badge variant="primary">{t("onboarding.recommended")}</Badge>
            )}
            {status === "active" && (
              <Badge variant="primary">
                <Check className="w-3 h-3 mr-1" />
                {t("modelSelector.active")}
              </Badge>
            )}
            {status === "loading" && (
              <Badge variant="secondary">
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                {t("modelSelector.switching")}
              </Badge>
            )}
          </div>
          <p
            className={`text-text/60 text-sm leading-relaxed ${isLocal ? "font-mono text-xs break-all" : ""}`}
          >
            {model.description}
          </p>
        </div>
      </div>

      <hr className="w-full border-mid-gray/20" />

      <div className="flex items-center gap-3 w-full -mb-0.5 mt-0.5 h-5">
        {model.parameters ? (
          <span
            className="flex items-center gap-1 text-xs text-text/50"
            title={t("settings.models.enhance.parametersTooltip")}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>{model.parameters}</span>
          </span>
        ) : (
          <span
            className="flex items-center gap-1 text-xs text-text/50"
            title={t("settings.models.enhance.quantTooltip")}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>{model.quant}</span>
          </span>
        )}
        <span
          className="flex items-center gap-1 text-xs text-text/50"
          title={t("settings.models.enhance.memoryTooltip")}
        >
          <MemoryStick className="w-3.5 h-3.5" />
          <span>
            {/* Formatted rather than rounded to whole gigabytes: the smallest
                models need well under 1 GB, and rounding rendered them as
                "0 GB RAM". */}
            {t("settings.models.enhance.memoryValue", {
              ram: formatModelSize(Number(model.min_ram_mb)),
            })}
          </span>
        </span>
        <span className="flex items-center gap-1.5 ms-auto text-xs text-text/50">
          {status === "downloadable" ? (
            <Download className="w-3.5 h-3.5" />
          ) : (
            <HardDrive className="w-3.5 h-3.5" />
          )}
          <span>{formatModelSize(Number(model.size_bytes) / 1_048_576)}</span>
        </span>
        {onDelete && (status === "available" || status === "active") && (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(model.id);
            }}
            title={
              isLocal
                ? t("settings.models.enhance.local.forgetTooltip")
                : t("modelSelector.deleteModel", { modelName: model.name })
            }
            className="flex items-center gap-1.5 text-logo-primary/85 hover:text-logo-primary hover:bg-logo-primary/10"
          >
            {isLocal ? (
              <X className="w-3.5 h-3.5" />
            ) : (
              <Trash2 className="w-3.5 h-3.5" />
            )}
            <span>
              {isLocal
                ? t("settings.models.enhance.local.forget")
                : t("common.delete")}
            </span>
          </Button>
        )}
      </div>

      {status === "downloading" && (
        <div className="w-full mt-3">
          <div className="w-full h-1.5 bg-mid-gray/20 rounded-full overflow-hidden">
            {downloadProgress !== undefined && downloadProgress > 0 ? (
              <div
                className="h-full bg-logo-primary rounded-full transition-all duration-300"
                style={{ width: `${downloadProgress}%` }}
              />
            ) : (
              // The transfer is starting and no byte count has arrived yet;
              // a bar stuck at zero reads as a failure.
              <div className="h-full bg-logo-primary rounded-full animate-pulse w-full" />
            )}
          </div>
          <div className="flex items-center justify-between text-xs mt-1">
            <span className="text-text/50">
              {downloadProgress !== undefined && downloadProgress > 0
                ? t("modelSelector.downloading", {
                    percentage: Math.round(downloadProgress),
                  })
                : t("settings.models.enhance.starting")}
            </span>
            {downloadSpeed !== undefined && downloadSpeed > 0 && (
              <span className="tabular-nums text-text/50">
                {t("modelSelector.downloadSpeed", {
                  speed: downloadSpeed.toFixed(1),
                })}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhanceModelCard;
