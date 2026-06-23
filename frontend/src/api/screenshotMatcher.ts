/**
 * Screenshot matcher — fuzzy-matches extracted names from vision API
 * against the game's item database to resolve item IDs.
 */

import type { Gun, ModInfo } from './client'

export interface MatchedWeapon {
  gun: Gun
  confidence: number
}

export interface MatchedMod {
  extractedName: string
  mod: ModInfo | null
  confidence: number
  locked: boolean
}

/**
 * Normalize a string for fuzzy comparison: lowercase, strip punctuation,
 * collapse whitespace.
 */
function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[''`"]/g, '')
    .replace(/[-_/\\().,:;!?]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Compute similarity between two strings using bigram overlap (Dice coefficient).
 * Returns 0–1 where 1 is identical.
 */
function bigramSimilarity(a: string, b: string): number {
  if (a === b) return 1
  if (a.length < 2 || b.length < 2) return 0

  const bigramsA = new Map<string, number>()
  for (let i = 0; i < a.length - 1; i++) {
    const bigram = a.slice(i, i + 2)
    bigramsA.set(bigram, (bigramsA.get(bigram) ?? 0) + 1)
  }

  const bigramsB = new Map<string, number>()
  for (let i = 0; i < b.length - 1; i++) {
    const bigram = b.slice(i, i + 2)
    bigramsB.set(bigram, (bigramsB.get(bigram) ?? 0) + 1)
  }

  let intersection = 0
  for (const [bigram, count] of bigramsA) {
    intersection += Math.min(count, bigramsB.get(bigram) ?? 0)
  }

  return (2 * intersection) / (a.length - 1 + b.length - 1)
}

/**
 * Check if one string contains the other as a substring.
 */
function containsMatch(query: string, target: string): boolean {
  return target.includes(query) || query.includes(target)
}

/**
 * Score a candidate item name against an extracted name.
 * Returns 0–1 confidence score.
 */
function scoreMatch(extracted: string, candidateName: string): number {
  const ne = normalize(extracted)
  const nc = normalize(candidateName)

  if (ne === nc) return 1.0

  // Substring match — scale confidence by how much of the candidate the
  // extracted text covers so generic single words like "stock" don't get 0.9
  // against "ak 12 stock".
  if (containsMatch(ne, nc)) {
    const shorter = Math.min(ne.length, nc.length)
    const longer = Math.max(ne.length, nc.length)
    const coverage = shorter / longer
    // Full-length match → 0.9, half-length → 0.6, quarter → 0.45
    return 0.5 + 0.4 * coverage
  }

  // Try matching against words — if extracted is a prefix/abbreviation
  const eWords = ne.split(' ')
  const cWords = nc.split(' ')
  let wordMatches = 0
  for (const ew of eWords) {
    if (cWords.some(cw => cw.startsWith(ew) || cw === ew)) {
      wordMatches++
    }
  }
  const wordScore = eWords.length > 0 ? wordMatches / eWords.length : 0
  if (wordScore >= 0.8) return 0.85

  const sim = bigramSimilarity(ne, nc)
  return sim
}

/**
 * Match an extracted weapon name against the available guns.
 */
export function matchWeapon(
  extractedName: string,
  guns: Gun[],
): MatchedWeapon | null {
  if (!extractedName) return null

  let bestGun: Gun | null = null
  let bestScore = 0

  for (const gun of guns) {
    const score = scoreMatch(extractedName, gun.name)
    if (score > bestScore) {
      bestScore = score
      bestGun = gun
    }
  }

  if (bestGun && bestScore >= 0.3) {
    return { gun: bestGun, confidence: bestScore }
  }
  return null
}

/**
 * Match extracted mod names against available mods for a weapon.
 * Returns matches sorted by extraction order. When multiple extracted names
 * resolve to the same mod, only the highest-confidence one is auto-locked
 * to avoid solver conflicts from duplicate forced items.
 */
export function matchMods(
  extractedNames: string[],
  availableMods: ModInfo[],
): MatchedMod[] {
  // First pass: collect top candidates per extracted name (not just the single best)
  const candidates = extractedNames.map(name => {
    const scored: { mod: ModInfo; score: number }[] = []

    for (const mod of availableMods) {
      const nameScore = scoreMatch(name, mod.name)
      const shortScore = mod.shortName ? scoreMatch(name, mod.shortName) : 0
      const score = Math.max(nameScore, shortScore)
      if (score >= 0.3) {
        scored.push({ mod, score })
      }
    }

    scored.sort((a, b) => b.score - a.score)
    return { extractedName: name, candidates: scored }
  })

  // Second pass: greedily pick the best candidate that doesn't conflict with
  // already-picked items.  This resolves ties like two "MOE SL" variants where
  // one conflicts with the locked barrel and the other doesn't.
  const pickedIds = new Set<string>()
  const pickedConflicts = new Set<string>()

  const raw: MatchedMod[] = candidates.map(({ extractedName, candidates: cands }) => {
    for (const { mod, score } of cands) {
      const modConflicts = new Set(mod.conflicting_item_ids ?? [])
      // Skip if this mod conflicts with an already-picked mod
      if (pickedConflicts.has(mod.id)) continue
      // Skip if an already-picked mod conflicts with this one
      let conflictsWithPicked = false
      for (const pid of pickedIds) {
        if (modConflicts.has(pid)) {
          conflictsWithPicked = true
          break
        }
      }
      if (conflictsWithPicked) continue

      // Pick this candidate
      pickedIds.add(mod.id)
      for (const cid of (mod.conflicting_item_ids ?? [])) {
        pickedConflicts.add(cid)
      }
      return {
        extractedName,
        mod,
        confidence: score,
        locked: score >= 0.8,
      }
    }

    // No non-conflicting candidate — fall back to best match (unlocked)
    const best = cands[0]
    return {
      extractedName,
      mod: best ? best.mod : null,
      confidence: best ? best.score : 0,
      locked: false,
    }
  })

  // Deduplicate: when multiple extractions match the same mod ID,
  // only lock the one with the highest confidence.
  const bestPerMod = new Map<string, number>()
  for (const m of raw) {
    if (!m.mod || !m.locked) continue
    const prev = bestPerMod.get(m.mod.id)
    if (prev === undefined || m.confidence > raw[prev].confidence) {
      bestPerMod.set(m.mod.id, raw.indexOf(m))
    }
  }
  for (let i = 0; i < raw.length; i++) {
    const m = raw[i]
    if (m.mod && m.locked && bestPerMod.get(m.mod.id) !== i) {
      raw[i] = { ...m, locked: false }
    }
  }

  // Resolve item conflicts: when two locked mods conflict with each other,
  // unlock the lower-confidence one to avoid infeasible solver models.
  for (let i = 0; i < raw.length; i++) {
    const a = raw[i]
    if (!a.mod || !a.locked) continue
    const aConflicts = new Set(a.mod.conflicting_item_ids ?? [])
    for (let j = i + 1; j < raw.length; j++) {
      const b = raw[j]
      if (!b.mod || !b.locked) continue
      const bConflicts = new Set(b.mod.conflicting_item_ids ?? [])
      if (aConflicts.has(b.mod.id) || bConflicts.has(a.mod.id)) {
        // Unlock the lower-confidence one
        if (a.confidence >= b.confidence) {
          raw[j] = { ...b, locked: false }
        } else {
          raw[i] = { ...a, locked: false }
          break
        }
      }
    }
  }

  return raw
}
