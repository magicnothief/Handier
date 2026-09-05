/**
 * Model ids that point at a file on disk rather than at the catalog.
 *
 * Mirrors `LOCAL_PREFIX` in `src-tauri/src/enhance/catalog.rs`. Kept here rather
 * than inlined at each use so the two places that build such an id and the three
 * that recognise one cannot drift apart.
 */
export const LOCAL_MODEL_PREFIX = "local:";

/** Whether `id` names a GGUF the user picked off disk. */
export function isLocalModelId(id: string): boolean {
  return id.startsWith(LOCAL_MODEL_PREFIX);
}

/** Build the model id that selects the GGUF at `path`. */
export function localModelId(path: string): string {
  return `${LOCAL_MODEL_PREFIX}${path}`;
}
