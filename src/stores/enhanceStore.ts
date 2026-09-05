import { create } from "zustand";
import { commands, events } from "@/bindings";
import { useSettingsStore } from "@/stores/settingsStore";
import type {
  EnhanceDownloadProgress,
  EnhanceModelInfo,
  EnhanceStatus,
} from "@/bindings";

interface DownloadStats {
  startTime: number;
  lastUpdate: number;
  lastBytes: number;
  speed: number; // MB/s
}

interface EnhanceStore {
  models: EnhanceModelInfo[];
  status: EnhanceStatus | null;
  /** Percentage per model id, for downloads in flight. */
  downloadProgress: Record<string, EnhanceDownloadProgress>;
  downloadStats: Record<string, DownloadStats>;
  /** Set while a model is being loaded into the sidecar. */
  loadingModelId: string | null;
  loading: boolean;
  initialized: boolean;
  error: string | null;

  initialize: () => Promise<void>;
  refresh: () => Promise<void>;
  downloadModel: (modelId: string) => Promise<void>;
  deleteModel: (modelId: string) => Promise<void>;
  /** Choose the editing model, or `null` to fall back to the catalog default. */
  selectModel: (modelId: string | null) => Promise<void>;
  setError: (error: string | null) => void;
}

/**
 * Enhancement models, their download state, and the sidecar's status.
 *
 * Deliberately a store rather than component state. The settings page used to
 * remember "this model is downloading" locally, so navigating away and back
 * lost it: the UI showed an idle download button while the backend was still
 * fetching, and pressing it again queued another gigabyte-scale transfer. State
 * that outlives the component has to live outside the component.
 */
export const useEnhanceStore = create<EnhanceStore>()((set, get) => ({
  models: [],
  status: null,
  downloadProgress: {},
  downloadStats: {},
  loadingModelId: null,
  loading: true,
  initialized: false,
  error: null,

  setError: (error) => set({ error }),

  refresh: async () => {
    const [status, models] = await Promise.all([
      commands.enhanceStatus(),
      commands.enhanceListModels(null),
    ]);
    if (status.status === "ok") set({ status: status.data });
    if (models.status === "ok") {
      set({ models: models.data, loading: false });
      // The backend is the authority on what is in flight. Adopting its view
      // is what lets a freshly mounted page show a download that started
      // before it existed.
      set((state) => {
        const progress = { ...state.downloadProgress };
        for (const model of models.data) {
          if (model.downloading && !progress[model.id]) {
            progress[model.id] = {
              modelId: model.id,
              downloaded: 0,
              total: Number(model.size_bytes),
              percentage: model.progress ?? 0,
              state: "running",
              error: null,
            };
          } else if (!model.downloading && progress[model.id]) {
            delete progress[model.id];
          }
        }
        return { downloadProgress: progress };
      });
    } else {
      set({ loading: false });
    }
  },

  initialize: async () => {
    if (get().initialized) return;
    set({ initialized: true });

    // Registered once for the lifetime of the app, not per mount, so no
    // progress is missed while the settings page is closed.
    void events.enhanceDownloadProgress.listen((event) => {
      const p = event.payload;
      set((state) => {
        const progress = { ...state.downloadProgress };
        const stats = { ...state.downloadStats };
        if (p.state === "running") {
          progress[p.modelId] = p;
          const now = Date.now();
          const prev = stats[p.modelId];
          if (!prev) {
            stats[p.modelId] = {
              startTime: now,
              lastUpdate: now,
              lastBytes: Number(p.downloaded),
              speed: 0,
            };
          } else {
            const seconds = (now - prev.lastUpdate) / 1000;
            // Below a second the sample is mostly jitter; keep the last figure
            // so the number on screen does not flicker.
            if (seconds >= 1) {
              const megabytes =
                (Number(p.downloaded) - prev.lastBytes) / 1_048_576;
              stats[p.modelId] = {
                ...prev,
                lastUpdate: now,
                lastBytes: Number(p.downloaded),
                speed: Math.max(0, megabytes / seconds),
              };
            }
          }
        } else {
          delete progress[p.modelId];
          delete stats[p.modelId];
        }
        return {
          downloadProgress: progress,
          downloadStats: stats,
          error: p.state === "failed" ? p.error : state.error,
        };
      });
      // A finished download changes what is on disk, so re-read the catalog.
      if (p.state !== "running") void get().refresh();
    });

    await get().refresh();
  },

  downloadModel: async (modelId) => {
    set({ error: null });
    // Show the download as started immediately. The first progress event only
    // arrives once the transfer is underway, and a button that looks unpressed
    // until then invites a second press.
    set((state) => ({
      downloadProgress: {
        ...state.downloadProgress,
        [modelId]: {
          modelId,
          downloaded: 0,
          total: 0,
          percentage: 0,
          state: "running",
          error: null,
        },
      },
    }));
    const result = await commands.enhanceDownloadModel(modelId);
    if (result.status === "error") {
      set((state) => {
        const progress = { ...state.downloadProgress };
        delete progress[modelId];
        return { downloadProgress: progress, error: result.error };
      });
    }
    await get().refresh();
  },

  deleteModel: async (modelId) => {
    set({ error: null });
    const result = await commands.enhanceDeleteModel(modelId);
    if (result.status === "error") set({ error: result.error });
    await get().refresh();
  },

  selectModel: async (modelId) => {
    set({ error: null, loadingModelId: modelId });
    try {
      const result = await commands.enhanceSetModel(modelId);
      if (result.status === "error") {
        set({ error: result.error });
        return;
      }
      // Nothing pushes a settings change back to the frontend, so the settings
      // store has to re-read it. Without this the page keeps showing the
      // previous card as active, because that is still what it thinks is set.
      await useSettingsStore.getState().refreshSettings();
    } finally {
      set({ loadingModelId: null });
      await get().refresh();
    }
  },
}));
