import { create } from "zustand";
import type { SidebarSection } from "@/components/Sidebar";

/** Anchors a component can ask to be scrolled to after a section change. */
export type NavAnchor = "enhancement-models";

interface NavStore {
  section: SidebarSection;
  /**
   * Where in the section to land, if anywhere.
   *
   * A section is not always one screenful. The Models page leads with the
   * transcription catalogue and puts the enhancement models below a rule, so
   * "Manage" in the enhancement settings used to drop the user at the top of a
   * long page with no sign of what they clicked for. Naming the destination
   * lets the section scroll to it once it has mounted.
   */
  anchor: NavAnchor | null;
  setSection: (section: SidebarSection, anchor?: NavAnchor) => void;
  /** Consume the anchor, so returning to the section later lands normally. */
  clearAnchor: () => void;
}

/**
 * Which settings section is on screen.
 *
 * Lifted out of `App` so a component can send the user somewhere else — the
 * enhancement settings need to point at the model page, and a button that only
 * *names* the page it wants is the kind of seam that makes a feature feel
 * bolted on.
 */
export const useNavStore = create<NavStore>()((set) => ({
  section: "general",
  anchor: null,
  setSection: (section, anchor) => set({ section, anchor: anchor ?? null }),
  clearAnchor: () => set({ anchor: null }),
}));
