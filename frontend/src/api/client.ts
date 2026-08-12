/**
 * API Client - uses Web Worker + HiGHS solver instead of HTTP backend.
 * All exported function signatures remain identical for UI compatibility.
 */

// --- Type exports (unchanged) ---

export interface Gun {
  id: string;
  name: string;
  image?: string;
  category: string;
  caliber: string;
  base_moa?: number;
}

export interface InfoResponse {
  guns: Gun[];
}

/** Solver LP mode: Fast (default omit), Precise, or Auto (precise when mod tree is small). */
export type SolverPrecisionMode = 'auto' | 'fast' | 'precise';

export interface OptimizeRequest {
  weapon_id: string;
  max_price?: number;
  min_ergonomics?: number;
  max_recoil_v?: number;
  max_recoil_sum?: number;
  min_mag_capacity?: number;
  min_sighting_range?: number;
  max_weight?: number;
  max_moa?: number;
  include_items?: string[];
  exclude_items?: string[];
  include_categories?: string[][];
  exclude_categories?: string[];
  ergo_weight?: number;
  recoil_weight?: number;
  price_weight?: number;
  trader_levels?: {
    prapor: number;
    skier: number;
    peacekeeper: number;
    mechanic: number;
    jaeger: number;
  };
  flea_available?: boolean;
  barter_available?: boolean;
  barter_exclude_dogtags?: boolean;
  /** Skip flea offers with <=3 active listings ("scarce") during price resolution. Default off. */
  exclude_scarce?: boolean;
  player_level?: number;
  /**
   * Completed-task IDs from a linked TarkovTracker account. Omitted/undefined
   * means unconfirmed — quest-gated trader offers stay available (flagged in
   * the UI) instead of being excluded. Only a confirmed absence excludes.
   */
  completed_task_ids?: string[];
  /** Boolean values are normalized: true → precise, false → fast. */
  precise_mode?: boolean | SolverPrecisionMode;
  /**
   * Force the build base: a preset item ID from the weapon's presets, or
   * 'naked' for the stock gun. Omitted = solver picks the optimal base.
   * If the forced base is unavailable at the given trader/flea settings, the
   * solver falls back to auto and flags it via `preset_unavailable_fallback`.
   */
  preset_id?: string;
}

export interface ItemDetail {
  id: string;
  name: string;
  price: number;
  icon?: string;
  source?: string;
  /** When false, part is not on traders/flea in API — optimizer assumes FiR/owned (buy cost 0) */
  purchasable?: boolean;
  /** BSG reference value for display when not purchasable */
  reference_price_rub?: number;
  ergonomics: number;
  recoil_modifier: number;
  barter_requirements?: Array<{ name: string; count: number; unit_price: number; icon?: string }>;
  /** Flea purchases only: <=3 active listings at fetch time */
  scarce?: boolean;
  /** Flea purchases only: price data older than 24h */
  stale?: boolean;
  /** Flea purchases only: price deviates >2.5x from the 24h average */
  price_unstable?: boolean;
  /** Display name of the quest gating the chosen trader offer, when it has one. */
  task_unlock_name?: string;
  /** 'verified' when a linked TarkovTracker account confirms the quest is done; 'unverified' otherwise. */
  task_locked_status?: 'unverified' | 'verified';
  // Tooltip extras
  image_large?: string;
  accuracy_modifier?: number;
  weight?: number;
  category?: string;
  handbook_categories?: string[];
  capacity?: number;
  sighting_range?: number;
}

export interface PresetDetail {
  id: string;
  name: string;
  price: number;
  items: string[];
  icon?: string;
  /** API source key for the offer used (e.g. trader id, fleaMarket) */
  source?: string;
  /** Human-readable seller for the preset price (e.g. "Prapor", "Flea Market") */
  purchase_label?: string;
  barter_requirements?: Array<{ name: string; count: number; unit_price: number; icon?: string }>;
  /** Flea purchases only: <=3 active listings at fetch time */
  scarce?: boolean;
  /** Flea purchases only: price data older than 24h */
  stale?: boolean;
  /** Flea purchases only: price deviates >2.5x from the 24h average */
  price_unstable?: boolean;
  /** Display name of the quest gating the chosen preset offer, when it has one. */
  task_unlock_name?: string;
  /** 'verified' when a linked TarkovTracker account confirms the quest is done; 'unverified' otherwise. */
  task_locked_status?: 'unverified' | 'verified';
  // Tooltip extras
  image_large?: string;
  caliber?: string;
  fire_rate?: number;
  fire_modes?: string[];
  default_ergo?: number;
  default_recoil_v?: number;
  default_recoil_h?: number;
  weight?: number;
  parts_count?: number;
}

export interface FinalStats {
  ergonomics: number;
  recoil_vertical: number;
  recoil_horizontal: number;
  total_price: number;
  total_weight: number;
  moa: number;
}

export interface OptimizeResponse {
  status: string;
  selected_items: ItemDetail[];
  selected_preset?: PresetDetail;
  fallback_base?: Record<string, unknown>;
  objective_value: number;
  reason?: string;
  final_stats?: FinalStats;
  solve_time_ms?: number;
  precision_request?: SolverPrecisionMode;
  precision_resolved?: 'fast' | 'precise';
  /** Slot-item pairs [slotId, itemId][] for EFTForge build export */
  slot_pairs?: [string, string][];
  /** True when a forced preset_id base was unavailable at current settings and the solver picked the base automatically. */
  preset_unavailable_fallback?: boolean;
}

export type GameMode = 'regular' | 'pve';

export interface ModInfo {
  id: string;
  name: string;
  shortName: string;
  category: string;
  /** BSG category id — must match solver `category_id` (not display name). */
  category_id: string;
  category_normalized: string;
  handbook_categories: string[];
  category_child_ids: string[];
  icon?: string;
  capacity?: number;
  accuracy_modifier?: number;
  /** Barrel-only: centerOfImpact × MOA_K (MOA units). Replaces weapon base_moa when installed. */
  base_moa?: number;
  /** Cheapest available price in RUB (0 if not purchasable). */
  price?: number;
  /** IDs of items that conflict with this mod. */
  conflicting_item_ids?: string[];
}

export interface ModCategoryOption {
  id: string;
  name: string;
}

export interface ExploreRequest extends OptimizeRequest {
  ignore: 'price' | 'recoil' | 'ergo';
  steps?: number;
}

export interface ExplorePoint {
  ergo: number;
  recoil_pct: number;
  recoil_v: number;
  recoil_h: number;
  price: number;
  selected_items: ItemDetail[];
  selected_preset?: PresetDetail;
  slot_pairs?: [string, string][];
  /** Full stats of this point's build (for the in-page build detail view). */
  final_stats?: FinalStats;
  status: string;
  solve_time_ms?: number;
  /** See OptimizeResponse.preset_unavailable_fallback */
  preset_unavailable_fallback?: boolean;
}

export interface ExploreResponse {
  points: ExplorePoint[];
  total_solve_time_ms?: number;
  precision_request?: SolverPrecisionMode;
  precision_resolved?: 'fast' | 'precise';
  /** True when any point fell back from a forced preset_id to auto base selection. */
  preset_unavailable_fallback?: boolean;
}

export interface GunsmithConstraints {
  min_ergonomics?: number;
  max_recoil_sum?: number;
  min_mag_capacity?: number;
  min_sighting_range?: number;
  max_weight?: number;
}

export interface GunsmithTask {
  task_name: string;
  weapon_id: string;
  weapon_name: string;
  weapon_image?: string;
  constraints: GunsmithConstraints;
  required_item_ids: string[];
  required_item_names: string[];
  /** Auto-added hosts when a required mod has no weapon slot and only one parent mod chain */
  implicit_required_item_ids?: string[];
  implicit_required_item_names?: string[];
  required_category_group_ids: string[][];
  required_category_names: string[][];
}

export interface GunsmithTasksResponse {
  tasks: GunsmithTask[];
}

// --- Worker Communication ---

let worker: Worker | null = null;
let messageId = 0;
const pendingRequests = new Map<number, { resolve: (value: unknown) => void; reject: (reason: unknown) => void }>();

function rejectAllPending(reason: Error): void {
  for (const [, pending] of pendingRequests) {
    pending.reject(reason);
  }
  pendingRequests.clear();
}

/** Terminate the worker and fail all in-flight requests (fatal worker failure). */
function resetWorker(reason: Error): void {
  rejectAllPending(reason);
  if (worker) {
    worker.terminate();
    worker = null;
  }
}

function getWorker(): Worker {
  if (!worker) {
    const w = new Worker(
      new URL('../solver/solver.worker.ts', import.meta.url),
      { type: 'module' }
    );
    w.onmessage = (event: MessageEvent<{ type: string; id: number; payload: unknown }>) => {
      const { type, id, payload } = event.data;
      const pending = pendingRequests.get(id);
      if (!pending) return;
      pendingRequests.delete(id);

      if (type === 'error') {
        pending.reject(new Error(payload as string));
      } else {
        pending.resolve(payload);
      }
    };
    w.onerror = (event) => {
      console.error('Worker error:', event);
      const msg = event.message || 'Web Worker failed';
      resetWorker(new Error(msg));
    };
    w.onmessageerror = (event) => {
      console.error('Worker message error:', event);
      resetWorker(new Error('Web Worker message error'));
    };
    worker = w;
  }
  return worker;
}

function sendWorkerMessage<T>(type: string, payload: Record<string, unknown>): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = messageId++;
    pendingRequests.set(id, {
      resolve: resolve as (value: unknown) => void,
      reject,
    });
    getWorker().postMessage({ type, id, payload });
  });
}

// --- Exported API functions (same signatures, local solver) ---

export const getInfo = async (gameMode: GameMode = 'regular', lang: string = 'en'): Promise<InfoResponse> => {
  return sendWorkerMessage<InfoResponse>('getInfo', { lang, gameMode });
};

export const getWeaponMods = async (weaponId: string, gameMode: GameMode = 'regular', lang: string = 'en'): Promise<{ mods: ModInfo[] }> => {
  return sendWorkerMessage<{ mods: ModInfo[] }>('getWeaponMods', { weaponId, lang, gameMode });
};

/** A purchasable preset of a weapon, priced at the caller's trader/flea settings. */
export interface WeaponPresetOption {
  id: string;
  name: string;
  image?: string | null;
  /** Cheapest price in RUB at the given settings. */
  price: number;
  /** API source key of the cheapest offer (trader normalizedName, 'fleaMarket', 'barter:*'). */
  source?: string | null;
  /** Human-readable seller label (e.g. "Prapor", "Flea Market"). */
  label?: string | null;
}

export interface WeaponBaseOptions {
  presets: WeaponPresetOption[];
  /** Stock/naked gun base option (always present; `available` mirrors the LP's naked-base rule). */
  naked: { price: number; source?: string | null; available: boolean };
}

export interface BaseAvailabilitySettings {
  trader_levels?: OptimizeRequest['trader_levels'];
  flea_available?: boolean;
  barter_available?: boolean;
  barter_exclude_dogtags?: boolean;
  /** Skip flea offers with <=3 active listings ("scarce") during price resolution. Default off. */
  exclude_scarce?: boolean;
  player_level?: number;
  /** See OptimizeRequest.completed_task_ids. */
  completed_task_ids?: string[];
  /** Banned base IDs (preset id or the weapon's own id for the naked base) — never auto- or manually-selectable. */
  exclude_items?: string[];
}

export const getWeaponPresets = async (
  weaponId: string,
  availability: BaseAvailabilitySettings = {},
  gameMode: GameMode = 'regular',
  lang: string = 'en',
): Promise<WeaponBaseOptions> => {
  return sendWorkerMessage<WeaponBaseOptions>('getWeaponPresets', { weaponId, availability, lang, gameMode });
};

export const optimize = async (request: OptimizeRequest, gameMode: GameMode = 'regular', lang: string = 'en'): Promise<OptimizeResponse> => {
  return sendWorkerMessage<OptimizeResponse>('optimize', { request, lang, gameMode });
};

export const explore = async (request: ExploreRequest, gameMode: GameMode = 'regular', lang: string = 'en'): Promise<ExploreResponse> => {
  return sendWorkerMessage<ExploreResponse>('explore', { request, lang, gameMode });
};

export const getGunsmithTasks = async (gameMode: GameMode = 'regular', lang: string = 'en'): Promise<GunsmithTasksResponse> => {
  return sendWorkerMessage<GunsmithTasksResponse>('getGunsmithTasks', { lang, gameMode });
};

export const getStatus = async (gameMode: GameMode = 'regular', lang: string = 'en'): Promise<{ timestamp: number }> => {
  return sendWorkerMessage<{ timestamp: number }>('getStatus', { lang, gameMode });
};

export const computeMOAFloor = async (weaponId: string, gameMode: GameMode = 'regular', lang: string = 'en'): Promise<{ floor: number }> => {
  return sendWorkerMessage<{ floor: number }>('computeMOAFloor', { weaponId, lang, gameMode });
};

/**
 * Wipe IndexedDB cache and terminate the worker (clearing its in-memory dataCache).
 * Caller should reload the page afterwards so stores re-fetch fresh data from tarkov.dev.
 * DB name kept in sync with `DB_NAME` in solver/dataService.ts.
 */
export const clearDataCache = async (): Promise<void> => {
  resetWorker(new Error('Cache cleared by user'));
  if (typeof indexedDB === 'undefined') return;
  await new Promise<void>((resolve) => {
    const req = indexedDB.deleteDatabase('tarkov-optimizer-cache');
    req.onsuccess = () => resolve();
    req.onerror = () => resolve();
    req.onblocked = () => resolve();
  });
};
