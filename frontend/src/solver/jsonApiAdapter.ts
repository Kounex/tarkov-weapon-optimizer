/**
 * JSON API fallback adapter for tarkov.dev.
 *
 * The GraphQL API (https://api.tarkov.dev/graphql) is the primary data source,
 * but it has had extended outages (HTTP 503 for every query). The maintainer's
 * recommended alternative is the JSON API (https://json.tarkov.dev), which is
 * structurally different from GraphQL:
 *   - items keyed by id in a dict, `types` array discriminator ("gun"/"mods"/"preset")
 *   - names are placeholders ("<id> Name") translated via separate overlay endpoints
 *   - trader offers live on `buyFromTrader` with trader ids, no flea `buyFor` entries
 *   - barters are a flat list indexed by offered item, required items are id+count only
 *
 * This module reshapes JSON-API responses into the exact GraphQL-shaped objects
 * that dataService's extract* functions already consume, so all downstream
 * solver code stays untouched.
 *
 * Flea market note: neither API exposes individual active listings. The best
 * available proxy for "cheapest current listing" is the item's `lastLowPrice`,
 * and availability is gated on `lastOfferCount > 0` AND the item NOT carrying
 * the `noFlea` type flag (flea-banned items can still report nonzero offer
 * counts). We synthesize a GraphQL-style flea `buyFor` offer from those fields
 * — items failing either check are treated as flea-unavailable (mirrored on
 * the GraphQL path in dataService).
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RawItem = Record<string, any>;

const JSON_API_BASE = 'https://json.tarkov.dev';

/** Image/scalar fields copied verbatim when present (downstream uses ?? chains). */
const SCALAR_FIELDS = [
  'id', 'normalizedName', 'types', 'basePrice', 'avg24hPrice', 'lastLowPrice', 'low24hPrice',
  'lastOfferCount', 'updated', 'minLevelForFlea', 'weight', 'width', 'height',
  'accuracyModifier', 'ergonomicsModifier', 'recoilModifier', 'conflictingSlotIds',
  'iconLink', 'image8xLink', 'image512pxLink', 'inspectImageLink', 'baseImageLink', 'gridImageLink',
] as const;

/** Weapon property fields read by extractGunStats (GraphQL ItemPropertiesWeapon). */
const GUN_PROP_FIELDS = [
  'caliber', 'effectiveDistance', 'sightingRange', 'fireRate', 'fireModes', 'cameraSnap',
  'centerOfImpact', 'deviationMax', 'deviationCurve', 'recoilAngle', 'recoilDispersion',
  'ergonomics', 'defaultErgonomics', 'recoilVertical', 'recoilHorizontal',
  'defaultRecoilVertical', 'defaultRecoilHorizontal',
] as const;

/** Mod property fields read by extractModStats (WeaponMod/Barrel/Magazine/Scope). */
const MOD_PROP_FIELDS = [
  'ergonomics', 'recoilModifier', 'accuracyModifier', 'centerOfImpact',
  'capacity', 'ammoCheckModifier', 'loadModifier', 'malfunctionChance',
  'sightingRange', 'sightModes', 'zoomLevels',
] as const;

interface JsonApiContext {
  items: Record<string, RawItem>;
  itemCategories: Record<string, RawItem>;
  handbookCategories: Record<string, RawItem>;
  traders: Record<string, { name: string; normalizedName: string }>;
  bartersByOfferedItem: Map<string, RawItem[]>;
  /** Translate an items-endpoint placeholder ("<id> Name", "MOD_SCOPE", handbook id). */
  tr: (placeholder: string | undefined | null) => string;
}

async function fetchJson(url: string): Promise<RawItem> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`JSON API ${url}: HTTP ${resp.status}`);
  return resp.json();
}

/** Optional overlay fetch — a missing language overlay must not kill the fallback. */
async function fetchOverlay(url: string): Promise<Record<string, string>> {
  try {
    const json = await fetchJson(url);
    return (json.data ?? {}) as Record<string, string>;
  } catch {
    return {};
  }
}

function makeTranslator(
  primary: Record<string, string>,
  fallback: Record<string, string>,
): (placeholder: string | undefined | null) => string {
  return (placeholder) => {
    if (!placeholder) return '';
    return primary[placeholder] ?? fallback[placeholder] ?? placeholder;
  };
}

/**
 * Build GraphQL-shaped `buyFor` offers from JSON `buyFromTrader` + flea fields.
 * Trader offer source = trader normalizedName ('prapor', ...), matching what the
 * GraphQL API put in `source` (downstream: getAvailablePrice traderLevels lookup,
 * ItemRow traderIcons). Flea offers are synthesized from lastLowPrice and only
 * exist when lastOfferCount > 0 (see module docstring).
 */
function buildBuyFor(raw: RawItem, ctx: JsonApiContext): RawItem[] {
  const buyFor: RawItem[] = [];
  for (const offer of raw.buyFromTrader ?? []) {
    const trader = ctx.traders[offer.trader];
    if (!trader) continue;
    const priceRUB = offer.priceRUB ?? offer.price ?? 0;
    if (priceRUB <= 0) continue;
    buyFor.push({
      currency: offer.currency ?? 'RUB',
      price: offer.price ?? priceRUB,
      priceRUB,
      source: trader.normalizedName,
      vendor: {
        name: trader.name,
        normalizedName: trader.normalizedName,
        minTraderLevel: offer.minTraderLevel ?? 1,
        buyLimit: offer.buyLimit ?? 0,
      },
    });
  }
  // Flea availability: `lastOfferCount > 0` alone is NOT reliable — noFlea
  // (flea-banned) items can still report a nonzero count. The noFlea type flag
  // takes precedence; dataService.fleaMarketSignals applies the same rules on
  // the GraphQL path.
  const types: string[] = raw.types ?? [];
  if (!types.includes('noFlea') && (raw.lastOfferCount ?? 0) > 0 && (raw.lastLowPrice ?? 0) > 0) {
    buyFor.push({
      currency: 'RUB',
      price: raw.lastLowPrice,
      priceRUB: raw.lastLowPrice,
      source: 'fleaMarket',
      vendor: { name: 'Flea Market', normalizedName: 'fleaMarket' },
    });
  }
  return buyFor;
}

/** Reshape JSON slot into GraphQL slot shape ({id, name, nameId, required, filters:{allowedItems:[{id}]}}). */
function adaptSlot(slot: RawItem, ctx: JsonApiContext): RawItem {
  // JSON slot `name` (when present) is an overlay key like "MOD_SCOPE"; otherwise
  // derive it from nameId ("mod_pistol_grip" → "MOD_PISTOL_GRIP").
  const nameKey = slot.name ?? (slot.nameId ? String(slot.nameId).toUpperCase() : undefined);
  return {
    id: slot.id,
    name: ctx.tr(nameKey) || (slot.nameId ?? ''),
    nameId: slot.nameId,
    required: slot.required ?? false,
    filters: {
      allowedItems: (slot.filters?.allowedItems ?? []).map((id: string) => ({ id })),
    },
  };
}

/** Resolve a preset item (types contains "preset") into the GraphQL preset shape. */
function adaptPreset(raw: RawItem, ctx: JsonApiContext): RawItem {
  const preset: RawItem = {
    id: raw.id,
    name: ctx.tr(raw.name),
    shortName: ctx.tr(raw.shortName),
    containsItems: (raw.containsItems ?? []).map((ci: RawItem) => ({
      item: { id: ci.item },
      count: ci.count ?? 1,
    })),
    buyFor: buildBuyFor(raw, ctx),
    bartersFor: ctx.bartersByOfferedItem.get(raw.id) ?? [],
  };
  // Copy all scalar fields (images + lastLowPrice/low24hPrice/lastOfferCount/
  // avg24hPrice/updated/types) — fleaMarketSignals in dataService reads them
  // off the preset for pricing, availability, and the unstable flag.
  for (const f of SCALAR_FIELDS) {
    if (f !== 'id' && raw[f] != null) preset[f] = raw[f];
  }
  return preset;
}

/**
 * Rebuild a GraphQL-shaped `bsgCategory` from JSON `categories` (a list of ids).
 * JSON gives multiple categories per item; GraphQL gave the single most specific
 * one, so we pick the deepest node in the parent chain. Parents are included two
 * levels up (buildCategoryPath walks them, skipping 'Compound item'/'Weapon mod').
 */
function adaptBsgCategory(categoryIds: string[] | undefined, ctx: JsonApiContext): RawItem | null {
  if (!categoryIds?.length) return null;

  const depthOf = (id: string): number => {
    let depth = 0;
    let node = ctx.itemCategories[id];
    const seen = new Set<string>();
    while (node?.parent && !seen.has(node.parent)) {
      seen.add(node.id);
      node = ctx.itemCategories[node.parent];
      depth++;
    }
    return depth;
  };

  let best: RawItem | undefined;
  let bestDepth = -1;
  for (const id of categoryIds) {
    const cat = ctx.itemCategories[id];
    if (!cat) continue;
    const depth = depthOf(id);
    if (depth > bestDepth) {
      bestDepth = depth;
      best = cat;
    }
  }
  if (!best) return null;

  const parent = best.parent ? ctx.itemCategories[best.parent] : undefined;
  const grandparent = parent?.parent ? ctx.itemCategories[parent.parent] : undefined;
  return {
    id: best.id,
    name: ctx.tr(best.name),
    normalizedName: best.normalizedName ?? '',
    parent: parent
      ? { name: ctx.tr(parent.name), parent: grandparent ? { name: ctx.tr(grandparent.name) } : null }
      : null,
    children: (best.children ?? []).map((id: string) => ({ id })),
  };
}

/** Shared scalar/category/offer reshaping for guns, mods, and presets' parents. */
function adaptItemBase(raw: RawItem, ctx: JsonApiContext): RawItem {
  const item: RawItem = {};
  for (const f of SCALAR_FIELDS) {
    if (raw[f] != null) item[f] = raw[f];
  }
  item.name = ctx.tr(raw.name);
  item.shortName = ctx.tr(raw.shortName);
  item.buyFor = buildBuyFor(raw, ctx);
  item.bartersFor = ctx.bartersByOfferedItem.get(raw.id) ?? [];
  item.conflictingItems = (raw.conflictingItems ?? []).map((id: string) => ({ id }));
  item.handbookCategories = (raw.handbookCategories ?? []).map((id: string) => ({
    // Handbook category names translate via the items overlay keyed by the raw id
    name: ctx.tr(ctx.handbookCategories[id]?.name ?? id),
  }));
  item.bsgCategory = adaptBsgCategory(raw.categories, ctx);
  return item;
}

function adaptGun(raw: RawItem, ctx: JsonApiContext): RawItem {
  const item = adaptItemBase(raw, ctx);
  const rawProps = raw.properties ?? {};
  const props: RawItem = {};
  for (const f of GUN_PROP_FIELDS) {
    if (rawProps[f] != null) props[f] = rawProps[f];
  }
  // fireModes may be overlay keys ("WEAPONFIREMODE_*"); translate, fall back to raw
  if (Array.isArray(props.fireModes)) {
    props.fireModes = props.fireModes.map((m: string) => ctx.tr(m));
  }
  props.slots = (rawProps.slots ?? []).map((s: RawItem) => adaptSlot(s, ctx));
  // JSON `presets` / `defaultPreset` are preset item id strings — resolve them
  props.presets = (rawProps.presets ?? [])
    .map((id: string) => ctx.items[id])
    .filter((p: RawItem | undefined): p is RawItem => Boolean(p))
    .map((p: RawItem) => adaptPreset(p, ctx));
  const defaultPresetRaw = rawProps.defaultPreset ? ctx.items[rawProps.defaultPreset] : undefined;
  if (defaultPresetRaw) {
    const dp: RawItem = {};
    for (const f of SCALAR_FIELDS) {
      if ((f.startsWith('icon') || f.includes('Image') || f.includes('image')) && defaultPresetRaw[f] != null) {
        dp[f] = defaultPresetRaw[f];
      }
    }
    props.defaultPreset = dp;
  }
  item.properties = props;
  return item;
}

function adaptMod(raw: RawItem, ctx: JsonApiContext): RawItem {
  const item = adaptItemBase(raw, ctx);
  const rawProps = raw.properties ?? {};
  const props: RawItem = {};
  for (const f of MOD_PROP_FIELDS) {
    if (rawProps[f] != null) props[f] = rawProps[f];
  }
  props.slots = (rawProps.slots ?? []).map((s: RawItem) => adaptSlot(s, ctx));
  item.properties = props;
  return item;
}

/**
 * Fetch guns + mods from the tarkov.dev JSON API, reshaped into the exact
 * GraphQL `items` entry shape that dataService's extract* functions consume.
 */
export async function fetchFromJsonApi(
  lang: string,
  gameMode: string,
): Promise<{ guns: RawItem[]; mods: RawItem[] }> {
  const mode = gameMode || 'regular';
  const needLangOverlay = lang !== 'en';

  const [itemsJson, itemsEnOverlay, itemsLangOverlay, tradersJson, tradersEnOverlay, tradersLangOverlay, bartersJson] =
    await Promise.all([
      fetchJson(`${JSON_API_BASE}/${mode}/items`),
      fetchOverlay(`${JSON_API_BASE}/${mode}/items_en`),
      needLangOverlay ? fetchOverlay(`${JSON_API_BASE}/${mode}/items_${lang}`) : Promise.resolve({}),
      fetchJson(`${JSON_API_BASE}/${mode}/traders`),
      fetchOverlay(`${JSON_API_BASE}/${mode}/traders_en`),
      needLangOverlay ? fetchOverlay(`${JSON_API_BASE}/${mode}/traders_${lang}`) : Promise.resolve({}),
      fetchJson(`${JSON_API_BASE}/${mode}/barters`),
    ]);

  const data = itemsJson.data ?? {};
  const rawItems: Record<string, RawItem> = data.items ?? {};
  const rawTraders: Record<string, RawItem> = tradersJson.data ?? {};
  const rawBarters: RawItem[] = bartersJson.data ?? [];

  const tr = makeTranslator(itemsLangOverlay, itemsEnOverlay);
  const tradersTr = makeTranslator(tradersLangOverlay, tradersEnOverlay);

  // Trader id → translated {name, normalizedName}
  const traders: JsonApiContext['traders'] = {};
  for (const [id, trader] of Object.entries(rawTraders)) {
    traders[id] = {
      name: tradersTr(trader.name) || trader.normalizedName || id,
      normalizedName: trader.normalizedName ?? '',
    };
  }

  // Index barters by offered item id → GraphQL `bartersFor` shape.
  // The barters endpoint only carries ids + counts, so required-item
  // prices/icons/names are looked up from the items dict. `lastLowPrice` is
  // included so extractBarterOffers can prefer current-lowest-offer pricing
  // over 24h averages (flea pricing realism).
  const bartersByOfferedItem = new Map<string, RawItem[]>();
  for (const barter of rawBarters) {
    const offeredId = barter.offeredItem?.item;
    const trader = traders[barter.trader];
    if (!offeredId || !trader) continue;
    const entry = {
      trader: { name: trader.name, normalizedName: trader.normalizedName },
      level: barter.minTraderLevel ?? 1,
      requiredItems: (barter.requiredItems ?? []).map((ri: RawItem) => {
        const reqItem = rawItems[ri.item];
        return {
          item: {
            id: ri.item,
            name: reqItem ? tr(reqItem.name) : ri.item,
            avg24hPrice: reqItem?.avg24hPrice ?? null,
            lastLowPrice: reqItem?.lastLowPrice ?? null,
            basePrice: reqItem?.basePrice ?? null,
            iconLink: reqItem?.iconLink ?? null,
          },
          count: ri.count ?? 1,
        };
      }),
    };
    const list = bartersByOfferedItem.get(offeredId);
    if (list) list.push(entry);
    else bartersByOfferedItem.set(offeredId, [entry]);
  }

  const ctx: JsonApiContext = {
    items: rawItems,
    itemCategories: data.itemCategories ?? {},
    handbookCategories: data.handbookCategories ?? {},
    traders,
    bartersByOfferedItem,
    tr,
  };

  const guns: RawItem[] = [];
  const mods: RawItem[] = [];
  for (const raw of Object.values(rawItems)) {
    const types: string[] = raw.types ?? [];
    if (types.includes('gun')) guns.push(adaptGun(raw, ctx));
    else if (types.includes('mods')) mods.push(adaptMod(raw, ctx));
  }
  return { guns, mods };
}
