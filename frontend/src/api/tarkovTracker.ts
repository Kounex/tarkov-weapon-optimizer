/**
 * TarkovTracker.org progress link — lets users prove quest completion so
 * quest-gated trader offers (e.g. a stock only sellable after "I Need More
 * Power") can be confirmed instead of just flagged.
 *
 * api.tarkovtracker.org locks CORS to its own origin, so the browser calls a
 * same-origin proxy route (vision-proxy sidecar, see AGENTS.md) that forwards
 * the user's own Bearer token as a pure passthrough — no server-side secret.
 */

const DEFAULT_PROXY_URL = import.meta.env.VITE_TARKOVTRACKER_PROXY_URL ?? '/api/tarkovtracker';

export interface TarkovTrackerProgress {
  displayName: string;
  /** IDs of tasks marked complete — tarkov.dev-style task IDs, matching OfferInfo.task_unlock. */
  completedTaskIds: string[];
}

interface ProgressTask {
  id: string;
  complete: boolean;
}

interface ProgressResponse {
  success: boolean;
  error?: string;
  data?: {
    displayName?: string;
    tasksProgress?: ProgressTask[];
  };
}

export async function fetchTarkovTrackerProgress(token: string): Promise<TarkovTrackerProgress> {
  const trimmed = token.trim();
  if (!trimmed) throw new Error('No TarkovTracker token configured');

  const response = await fetch(`${DEFAULT_PROXY_URL}/progress`, {
    headers: { Authorization: `Bearer ${trimmed}` },
  });

  const body = await response.json().catch(() => null) as ProgressResponse | null;
  if (!response.ok || !body?.success || !body.data) {
    throw new Error(body?.error ?? `TarkovTracker API error (${response.status})`);
  }

  const completedTaskIds = (body.data.tasksProgress ?? [])
    .filter(t => t.complete)
    .map(t => t.id);

  return {
    displayName: body.data.displayName ?? '',
    completedTaskIds,
  };
}
