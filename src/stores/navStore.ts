import { create } from "zustand";
import type { SidebarSection } from "@/components/Sidebar";

interface NavStore {
  section: SidebarSection;
  setSection: (section: SidebarSection) => void;
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
  setSection: (section) => set({ section }),
}));
