# Changelog

All notable changes to the Tarkov Weapon Mod Optimizer.

## [v2.7.4] — 2026-08-13

### Fixed
- **Banning a base preset or the naked receiver showed an empty, unlabeled red chip in the Mod Filter's exclude list.** Base bans share the same underlying list as mod bans (`excludedModIds`), but `ModFilter` only ever knew how to resolve mod names — a preset/weapon id isn't in `availableMods`, so the chip rendered with no text. `ModFilter` now only renders entries it can actually resolve to a mod; banned bases get their own chip list next to the "Base preset" selector in `WeaponSelector` instead, resolved from the weapon's own name (naked) or the already-cached preset name lookup (a preset can only be banned after being shown in a build result, so its name is always available by then).

## [v2.7.3] — 2026-08-13

### Added
- **Ban a base preset or the naked/stock receiver from being used at all.** The build result's base-preset card now has the same ban action mods already have (`UsingPresetCard`), so a base you don't actually have access to — quest-locked or otherwise — can be permanently excluded rather than just flagged. Reuses the exact same ban list as mods: banning a base's ID (the preset's own item ID, or the weapon's own ID for the naked receiver) removes it from auto-selection (`lpBuilder.ts`) and from the manual "Base preset" dropdown (`getWeaponPresets`) alike. A banned-but-forced base falls back to auto selection with the existing "base unavailable" warning, same as any other unpurchasable forced base. i18n in all 16 locales.

## [v2.7.2] — 2026-08-13

### Fixed
- **Stale cached game data kept showing the v2.7.1 quest-name bug after the fix shipped.** The v2.7.1 fix changed what's stored inside the cached game-data blob (resolved quest names) without bumping `CACHE_VERSION`, so browsers with data already cached (1h TTL) kept serving the pre-fix names until the cache expired or "force refresh data" was used. `CACHE_VERSION` 17 → 18 invalidates every existing cache immediately.
- **The naked/stock gun base never went through the same availability filtering as mods and presets.** `extractGunStats` computed its price from the raw `buyFor` minimum, ignoring trader-level settings entirely, hardcoding flea market out of consideration ("naked gun is only purchasable via direct trader offers" — a deliberate but overly narrow carryover from the original Python CP-SAT model), and having no concept of quest-gated offers at all. It's now built the same per-offer way as mods/presets (trader offers incl. `taskUnlock`, a flea offer, barters) and resolved through `getAvailablePrice`, so: trader-level settings are finally respected for the naked base, flea is a valid fallback source when enabled and cheaper/available, quest-locked naked-gun offers get the same "Quest-locked" badge (and only get excluded once a linked TarkovTracker account confirms the quest isn't done) as mods and presets already did.

## [v2.7.1] — 2026-08-13

### Fixed
- **Quest-locked badge showed the raw quest ID instead of its name.** `jsonApiAdapter.ts` read the `/tasks` JSON-API response as a flat id→task dict, but that endpoint actually bundles several collections under `data` (`tasks`, `questItems`, `achievements`, `prestige`) with the real task dict nested one level deeper at `data.tasks`. Every lookup missed, silently falling back to the task ID string. Affects the JSON-API fallback path only (currently the active path — tarkov.dev's GraphQL API has been down since 2026-07-21); the primary GraphQL path already requested `taskUnlock { id name }` inline and was unaffected. Unrelated to whether a TarkovTracker account is linked.

## [v2.7.0] — 2026-08-13

### Added
- **TarkovTracker.org quest-completion link.** Trader offers gated behind a quest (independent of trader level — e.g. the AR-15 Magpul PRS GEN3 stock, only sellable at Peacekeeper LL3 after "I Need More Power") are no longer silently treated as unconditionally available. By default such an offer still shows (never excluded on a guess) but carries an orange "Quest-locked" badge naming the quest, so a build's real-life availability makes sense even without linking anything. Linking a read-only TarkovTracker API token in Settings → Player & Trader Access → TarkovTracker fetches your actual quest progress; only then does an offer get excluded from builds, and only when the link confirms the quest is genuinely incomplete. Flea/other trader offers for the same item are unaffected — exclusion is per-offer, not per-item. i18n in all 16 locales.
- `taskUnlock` data is now fetched on both the GraphQL and JSON-API fallback paths (the JSON path resolves quest names via a new `tasks`/`tasks_en` overlay fetch, matching the GraphQL shape).

## [v2.6.6] — 2026-07-29

### Added
- **Lock / ban mods from the Explore build-detail modal.** The modal's build manifest (Detailed/Compact/Table views) now has the same lock (must-include) and ban (never-use) row actions as the Optimize tab. Toggles update the shared include/exclude constraints immediately and rows show their locked/banned state; constraints take effect on the next manually triggered explore run (no auto-re-run of the whole Pareto frontier). Optimize and Explore share one include/exclude state, so constraints set in either tab apply to both.

## [v2.6.5] — 2026-07-28

### Added
- **Full in-page build view for Explore frontier builds.** Each build in the Explore results list now has a **View** button opening a modal with the exact Optimize-tab build rendering — stat cards (ergo/recoil/MOA/weight/total cost), preset card, and the full build manifest with item images, prices, trader icons, barter tooltips, and scarce/stale/unstable badges (Detailed/Compact/Table view modes included). Components are shared with the Optimize tab, not forked; the EFT Forge external link remains as a secondary action. i18n in all 16 locales.

## [v2.6.4] — 2026-07-28

### Added
- **"Exclude scarce offers" switch** in the Market & Trader Access box (shown when flea access is on, persisted with the other market settings). When enabled, the optimizer skips flea offers with ≤ 3 active listings — the same threshold as the "scarce" badge introduced in v2.6.2. Scarcity is evaluated per offer, not per item: a part still buyable from a trader (within trader-level settings) or via an enabled barter stays available; only parts whose *sole* source is a scarce flea listing drop out of builds entirely. Applies to Optimize, Explore, Gunsmith, and the base-preset listing. i18n in all 16 locales.

## [v2.6.3] — 2026-07-28

### Added
- **Base preset selector.** Once a weapon is selected, a "Base preset" dropdown lists every purchasable preset for it (image, name, live price, and source — trader/barter/flea at the current availability settings) plus **Stock (naked)** and the default **Auto** (solver picks the optimal base, unchanged behavior). Picking a preset forces the solver to build on top of exactly that base; the constraint applies to both Optimize and Explore modes. If the forced base isn't purchasable at the current trader/flea/barter settings, the solver falls back to auto and a warning toast explains why. i18n in all 16 locales.

### Fixed
- Weapon change now also clears the previously fetched preset list — the dropdown briefly showed the *previous* weapon's presets until the new fetch landed.

## [v2.6.2] — 2026-07-28

### Fixed
- **Flea-banned (`noFlea`) items are no longer treated as flea-available.** 53 mods (REAP-IR thermal, SureFire 60-rounders, …) carry tarkov.dev's `noFlea` flag and report a nonzero `lastOfferCount` despite being unlistable — so offer count alone is not a trustworthy availability signal. Both the GraphQL and JSON API paths now skip flea offers for `noFlea` items explicitly (previously only saved accidentally by their `lastLowPrice` being null). `types` is now fetched/passed through on both paths; `CACHE_VERSION` 16 → 17.

### Added
- **Bait-listing guard for flea prices.** Effective flea price is now `max(lastLowPrice, low24hPrice)` on both paths: a current listing priced below the day's lowest observed price is treated as a possible bait/outlier and priced conservatively instead of skewing build totals.
- **Flea price badges in build results.** Item rows (and preset cards) whose price comes from the flea market now show small badges: **scarce** (≤ 3 active offers at last scan), **stale** (price data older than 24 h), and **unstable** (current price deviates > 2.5× from the 24 h average — flag only, never changes the price used). With tooltips, in all 16 locales.

## [v2.6.1] — 2026-07-28

### Added
- **JSON API fallback for game data** (`jsonApiAdapter.ts`). The tarkov.dev GraphQL API has had extended outages (HTTP 503 on every query since 2026-07-21, upstream issue the-hideout/tarkov-api#474). On any GraphQL failure the app now falls back to the maintainer-recommended JSON API (`json.tarkov.dev`), reshaping its responses (id-keyed item dicts, placeholder names + per-language translation overlays, trader-id offers, flat barter list) into the GraphQL item shape the solver already consumes — no downstream changes. GraphQL stays the primary source; fallback results are cached in IndexedDB like GraphQL results (`CACHE_VERSION` 15 → 16). GraphQL retries reduced from 3 (~14s worst case) to 2 quick attempts so the fallback kicks in fast.

### Fixed
- **Flea-market prices now track current listings instead of 24h averages.** Neither tarkov.dev API exposes individual active listings, so flea offers are now priced at the item's `lastLowPrice` (current cheapest listing) on both the GraphQL and JSON paths, and items reporting `lastOfferCount <= 0` are treated as flea-unavailable (previously an average-based flea price was used even when nothing was actually listed). Barter required-item unit costs now prefer `lastLowPrice` over `avg24hPrice`. `avg24hPrice` remains only in reference/display roles.

## [v2.5.5] — 2026-04-20

### Added
- **Force-refresh data button** (top-right reload icon, mobile menu entry): clears the browser's IndexedDB game-data cache and terminates the in-memory solver worker, then reloads the page so all data is re-fetched from tarkov.dev. Useful when a stale cache produces odd builds or when the project wants to pick up fresh wipe data without waiting for the 1-hour TTL to expire. Translations added for all 16 locales.

### Fixed
- **Performance preset weights changed from (50, 50, 0) to (48, 48, 2)** to break objective-function degeneracy. With `price_weight = 0`, any two builds that only differ in expensive negative-ergo / low-recoil parts (e.g. M203 + 20" barrel vs lighter layouts) had identical objective values after tie-breaker, so HiGHS could return wildly different solutions across WASM runtimes (observed: Chromium → 46.5 ergo / 47.4 recoil V, Firefox → 0.5 ergo / 40.2 recoil V on the same M4A1 LP). The 2% price nudge is small enough to preserve the "recoil + ergo" intent but large enough (≈ ₽7M swing on typical builds) to pick a single deterministic optimum.

## [v2.5.4] — 2026-04-20

### Fixed
- **Optimization no longer fails with `Unable to read LP model` on most assault rifles** (AK-74, M4A1, AKM, AK-74M, and any weapon whose reachable-mod tree contains a self-referencing conflict). Tarkov.dev's data lists a few items in their own `conflicting_items` array, so the LP builder emitted `conf_N: x_i + x_i <= 1`, and HiGHS's CPLEX LP reader rejects duplicate columns in a constraint row, aborting the whole parse. Added a `cid === iid` guard in the conflict loop so self-conflicts are silently dropped. Smaller rifles (e.g. AKS-74U) coincidentally avoided the bug because the self-conflicting item wasn't reachable from them. Unrelated to the v2.5.x MOA work — latent since the conflict loop was introduced.

## [v2.5.3] — 2026-04-19

### Fixed
- **MOA values now match in-game display**. The conversion factor from BSG's internal `centerOfImpact` units to displayed MOA was wrong: we multiplied by 100, but the correct empirical factor is ≈ **34.3** (the widely-cited Tarkov community constant). Final MOA values were therefore ~2.9× too high — e.g. VPO-215 with its long/short barrels showed 4.10/6.00 on the site vs the real in-game 1.55/2.06; M700 barrels showed 1.41–2.35 vs the real sub-1. Added `MOA_K` constant in `lpBuilder.ts`, updated the LP big-M constraints (per-barrel + fallback + fixed-barrel), the final-stats display formula, and the weapon/barrel `base_moa` fields returned by the worker.
- Reverted a prior incorrect hypothesis that barrel COI *adds* to weapon intrinsic COI — BSG actually *replaces* when a replaceable barrel is installed, so `effectiveBaseCOI = barrelCOI > 0 ? barrelCOI : weaponCOI`.
- **Max MOA slider's displayed min no longer renders an infeasible cap**. Previously the exact floor was rounded to 2 decimals with `Math.round` — e.g. M700's true floor 0.4836 displayed as 0.48, but setting `maxMOA = 0.48` was infeasible. Switched to `Math.ceil` so the slider's leftmost position is always reachable (costs +0.005 MOA of display precision). Removed a re-flooring pass inside `WeightAdjuster` that undid the ceiling.

### Changed
- Result stats card now shows MOA with 3 decimal places (was 2), so small accuracy differences between mod choices are visible.
- Removed the orange base-MOA tick from the Max MOA slider — for weapons with replaceable barrels the intrinsic value isn't reachable anyway, and showing it cluttered the slider.

## [v2.5.2] — 2026-04-18

### Fixed
- **Exact MOA floor now correctly reports the true minimum** for weapons like M16A1/ADAR where accuracy-boosting mods exist deep in the slot graph. Previously the floor-finder used linear cap-tightening (0.005 per step, 12 iterations) which stalled well above the true minimum on weapons where the seed solve starts far from the floor. Switched to binary search between `[0, seed]`, converging in ~log₂ iterations — M16A1 now reports 2.34 instead of 3.0.
- **Max MOA constraint is now correctly enforced when the LP skips all COI-barrel mods**. Previously, for weapons where the barrel slot is optional and replacement barrels carry a `centerOfImpact`, the big-M constraints only bound *when* a COI-barrel was installed. If the LP chose to skip every COI-barrel (using the intrinsic COI instead), the constraint became vacuous and the user's cap could be silently violated. Added a fallback constraint using the weapon's intrinsic COI, guarded by the sum of barrel indicator variables so it only binds when no COI-barrel is installed.

## [v2.5.1] — 2026-04-18

### Fixed
- **Docker image no longer restart-loops on IPv6-enabled hosts**: nginx now listens on both IPv4 (`listen 80`) and IPv6 (`listen [::]:80`), and the container's HEALTHCHECK probes `127.0.0.1` instead of `localhost` to avoid IPv6 resolution ambiguity. Previously, on Docker daemons with IPv6 enabled, the healthcheck could resolve `localhost` to `::1`, fail against the v4-only nginx, mark the container unhealthy, and trigger a restart loop.
- Removed unused `constraints.reset` key from 14 non-en/zh locale files.

## [v2.5.0] — 2026-04-18

### Added
- **Max MOA constraint now honors replaceable barrels**: For weapons like the VPO-215 and M700, swapping to a different barrel correctly changes the displayed MOA and the hard-constraint limit uses the installed barrel's `centerOfImpact` (not the weapon's intrinsic one). LP uses an exact big-M per-barrel formulation.
- **Exact slider floor toggle** (new "精确下限" / "Exact slider floor" row under Max MOA): when on, the slider's minimum is computed by actually solving for the lowest achievable MOA (respects slot reachability, conflicts, and barrel-specific compatibility). When off, the minimum is a theoretical per-category estimate that may not be reachable in practice. Toggle state persists to localStorage.

### Fixed
- Slider range no longer includes the weapon's intrinsic `centerOfImpact` for weapons with a required replaceable-barrel slot — the intrinsic value is never reachable in those cases.
- Displayed final MOA now correctly reflects the installed barrel's COI instead of always using the weapon's intrinsic value.

## [v2.4.3] — 2026-04-17

### Added
- **Docker self-hosting support**: Multi-architecture (`linux/amd64`, `linux/arm64`) Docker images are now built and published to GitHub Container Registry on every release tag. Pull with `docker pull ghcr.io/ahaimk01/tarkov-optimizer-frontend:latest` — see README for the full quick-start recipe.
- Dockerfile and nginx config in `frontend/` support SPA fallback routing, correct WASM MIME handling, and tiered caching (immutable for hashed assets and `.wasm`, no-cache for locales and CHANGELOG).

## [v2.4.2] — 2026-04-12

### Fixed
- Language selector flags now render as SVG images (via `flag-icons`) instead of Unicode emoji, fixing broken display on Windows Chrome.
- Ternary plot (triangle weight picker) click and drag now works correctly in Chrome — dot no longer follows cursor on hover, only responds to click/drag.
- Fixed broken JSON syntax in 14 locale files (missing commas in constraints section).

## [v2.4.1] — 2026-04-12

### Added
- Mod categories now use **handbook categories** from the Tarkov.dev API, showing proper pluralized names (e.g. "Suppressors", "Magazines") instead of raw BSG category paths.
- Build result items display full handbook hierarchy path (e.g. "Weapon parts & mods > Functional mods > Suppressors") with leaf name shown in compact mode.

### Changed
- ModFilter category dropdown now uses antd built-in search filtering with inline +/- buttons per option.

### Fixed
- Fixed "Exclude dogtag barters" checkbox having no effect — the `barterExcludeDogtags` parameter was not passed to `getAvailablePrice` during LP construction.

## [v2.4.0] — 2026-04-11

### Added
- New **Table View** mode for results, providing a high-density spreadsheet layout for power users.
- Persistent view state (Detailed/Compact/Table) saved to local storage.
- Interactive Lock/Ban buttons directly inside the result table.
- New i18n keys for stats labels, units, and tooltips across the app.

### Changed
- Refactored mobile item cards for improved readability and vertical alignment.
- Updated accuracy (Acc) tag colors to Orange to distinguish them from Recoil (Green).
- Global footer now uses bullet separators and flex-wrapping to prevent layout orphans.

### Fixed
- Fixed missing `useTranslation` import in manifest components.
- Removed nested scrollbars in result containers to restore clean native browser scrolling.
- Locked price/trader columns in table mode to prevent horizontal layout shift.
- Translated ~25 untranslated UI keys (table headers, tooltips, lock/ban labels, barter labels) across all 14 non-English locales.
- Translated `ui.on`/`ui.off` toggles for all locales (was showing English "On"/"Off").
- Added missing `trader` section to 14 locale files.
- Fixed ternary plot vertex label showing preset name ("纯后坐") instead of axis label ("后坐").
- Fixed zh.json explore constraint labels still in English (limit_price, limit_recoil, limit_ergo).
- Removed unused imports (Tooltip, Switch, Select) causing TypeScript build failures.

## [v2.3.1] — 2026-04-10

### Fixed
- Fixed bug where tags would overlap horizontally with UI items when displayed in single-column on extremely small screens
- Safely drop to single-column earlier on narrow displays to prevent component squishing

## [v2.3.0] — 2026-04-10

### Added
- MOA stat card in optimize and gunsmith results — computed from weapon center of impact + mod accuracy modifiers
- Max MOA (Spread) hard constraint with weapon-specific slider range (best/base/worst marks)
- Accuracy (Acc) and sighting range tags on item rows alongside ergo/recoil
- Lock/ban buttons on result items to quickly require or exclude mods in next optimization
- Card-based item layout in detailed mode with 64x64 item icon, trader portrait, weight tag, price tag
- Retained items section moved into preset card as collapsible
- Category path shown under item name (e.g. "Gear mod > Magazine")
- "Hard Constraints" as a separate collapsible panel

### Changed
- Renamed "Level Config" to "Market & Trader Access"
- "Using Preset" label now shows "Naked Receiver" for naked gun builds
- Compact mode renders as slim single-line rows with horizontal lock/ban buttons

## [v2.2.1] — 2026-04-10

### Added
- Item tooltips with hi-res image, category path, weight, and capacity
- Preset tooltips with large weapon image
- Accuracy (MOA) and sighting range shown as inline tags on item rows
- Weight column in item rows
- Category path shown under item name (e.g. "Gear mod > Magazine")
- Lock/ban buttons on result items to quickly require or exclude mods in next build
- Two-column item grid layout on non-mobile screens
- Column divider line between item grid columns

### Changed
- Redesigned preset card — compact row layout with trader icon
- Use transparent-background images (image512px) instead of dark-bg icons
- Disabled light themes and auto mode — dark-only for now (item images need dark backgrounds)
- Default theme set to Dark OneDark

### Fixed
- Preset-retained items no longer show misleading trader source and price

## [v2.2.0] — 2026-04-10

### Added
- Barter trade support — toggle "Barter Trades" in Level Config to include barter-only items in optimization
- Barter cost calculated as flea market value of required trade-in items
- Barter source indicator — gold "B" badge on trader icons for barter-sourced items
- Barter requirement tooltip — hover to see trade-in items, counts, and flea prices
- Barter support for weapon presets (preset-only barters now selectable)

### Changed
- Preset-retained items no longer show a misleading trader source and price — they display "—" since they're included with the preset
- Version tag in header now shows full version (v2.2.0) instead of just "v2"

### Fixed
- Weapon search now accepts spaces (merged from community PR #5)

## [v2.1.2] — 2026-04-07

### Fixed
- Min magazine capacity defaults to Off instead of 30
- Switching weapons resets min magazine capacity to Off

## [v2.1.1] — 2026-04-07

### Changed
- Min magazine capacity: merged toggle switch into slider with "Off" tick at 0
- Synced missing i18n keys across all 16 locale files
- Added i18n sync step to release workflow in CLAUDE.md

## [v2.1.0] — 2026-04-07

### Added
- Min magazine capacity constraint with ticked slider showing valid capacities per weapon
- EFTForge export button for each Pareto frontier result in Explore tab
- Axis labels and uniform tick spacing on Explore scatter chart
- Optional budget limit for the ignored variable in Explore (e.g. cap price while exploring Ergo vs Recoil)
- Changelog modal accessible from footer

### Changed
- EFTForge buttons now open builds directly via URL parameter (`?build=<code>`) instead of clipboard
- Explore tradeoff labels from "Ignore X" to "X optional"
- Version management with semantic versioning (v2.1.0)

## [v2.0.0] — 2026-04-06

### Added
- **Web app** — full browser-based solver using HiGHS WASM, deployed to GitHub Pages (no backend required)
- EFTForge integration — export optimized builds with one click
- Solver precision mode (fast / precise / auto)
- Weight presets (pure recoil, pure ergo, balanced, budget, performance, recoil focus, ergo focus)
- Ternary plot for 3-way weight adjustment with slider toggle
- Dark / light / AMOLED theme support
- PvP / PvE game mode toggle
- 16 language support (en, ru, zh, es, de, fr, it, ja, ko, pl, pt, tr, cs, hu, ro, sk)

### Changed
- Solver ported from Python OR-Tools CP-SAT to HiGHS WASM LP (runs entirely in browser)
- Migrated to React 19 + Ant Design v6
- UI overhaul across all tabs

### Fixed
- Naked gun pricing (price=0 bug)
- LP builder numerical stability and FiR mod availability
- Solver buy-variable price accounting
- HiGHS WASM memory and gzip streaming corruption
- Trader icons and favicon path resolution
