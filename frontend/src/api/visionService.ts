/**
 * Vision service — sends screenshots to a Cloudflare Worker proxy that calls Gemini Flash.
 *
 * The Gemini API key is stored server-side in the Worker secret, never exposed to the client.
 * Falls back to direct Gemini API call if the user provides their own key (for local dev).
 */

const VISION_PROXY_URL_KEY = 'visionProxyUrl'
const GEMINI_API_KEY_STORAGE_KEY = 'geminiApiKey'

const DEFAULT_PROXY_URL = import.meta.env.VITE_VISION_PROXY_URL ?? '/api/vision'

export function getProxyUrl(): string {
  return localStorage.getItem(VISION_PROXY_URL_KEY) ?? DEFAULT_PROXY_URL
}

export function setProxyUrl(url: string): void {
  if (url) {
    localStorage.setItem(VISION_PROXY_URL_KEY, url)
  } else {
    localStorage.removeItem(VISION_PROXY_URL_KEY)
  }
}

export function getStoredApiKey(): string {
  return localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) ?? ''
}

export function setStoredApiKey(key: string): void {
  if (key) {
    localStorage.setItem(GEMINI_API_KEY_STORAGE_KEY, key)
  } else {
    localStorage.removeItem(GEMINI_API_KEY_STORAGE_KEY)
  }
}

export function isConfigured(): boolean {
  return !!(getProxyUrl() || getStoredApiKey())
}

export interface VisionExtractionResult {
  weapon_name: string
  mod_names: string[]
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      const base64 = result.split(',')[1]
      resolve(base64)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

/**
 * Extract weapon + mod names from a screenshot.
 *
 * Tries the proxy first (no API key needed). Falls back to direct Gemini call
 * if a user-provided API key is available.
 */
export async function extractFromScreenshot(file: File): Promise<VisionExtractionResult> {
  const base64 = await fileToBase64(file)
  const mimeType = file.type || 'image/png'

  const proxyUrl = getProxyUrl()
  if (proxyUrl) {
    return extractViaProxy(proxyUrl, base64, mimeType)
  }

  const apiKey = getStoredApiKey()
  if (apiKey) {
    return extractViaGeminiDirect(apiKey, base64, mimeType)
  }

  throw new Error('No vision service configured — set a proxy URL or Gemini API key')
}

async function extractViaProxy(
  proxyUrl: string,
  base64: string,
  mimeType: string,
): Promise<VisionExtractionResult> {
  const response = await fetch(`${proxyUrl}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64, mimeType }),
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText })) as { error?: string }
    throw new Error(body.error ?? `Proxy error (${response.status})`)
  }

  const data = (await response.json()) as VisionExtractionResult
  return {
    weapon_name: data.weapon_name ?? '',
    mod_names: Array.isArray(data.mod_names) ? data.mod_names : [],
  }
}

const EXTRACTION_PROMPT = `You are analyzing a screenshot of the weapon inspect screen from the game "Escape from Tarkov".

Your task: identify the weapon and ALL installed modification parts visible in the screenshot.

The inspect screen shows:
- The weapon name at the top (in the search/title bar)
- A rendered image of the assembled weapon
- Stats (ergonomics, recoil, accuracy, etc.)
- A row of installed mod/part icons at the bottom of the inspect panel, each showing a SHORT NAME label

IMPORTANT: Only extract actual mod/part short names from the bottom strip. Do NOT include:
- Generic slot category labels like "Stock", "Foregrip", "Mount", "Chamber", "Barrel", "Handguard", "Scope"
- The weapon's own name or receiver name (e.g. "M4A1", "AK-74")
- Stats text or UI labels

Mod short names are typically specific product names/abbreviations like "MAG5-60", "MOE SL", "WarComp", "MK12", "A2", "DS150 FDE", "ROMEO7", "HAMR".

Extract:
1. The full weapon name (e.g. "SVDS 7.62x54R semi-automatic sniper rifle")
2. Every specific mod/part short name visible in the bottom tab strip

Return ONLY valid JSON in this exact format, no markdown, no explanation:
{"weapon_name": "full weapon name here", "mod_names": ["mod1 short name", "mod2 short name", ...]}

If you cannot identify the weapon or mods, return:
{"weapon_name": "", "mod_names": []}
`

async function extractViaGeminiDirect(
  apiKey: string,
  base64: string,
  mimeType: string,
): Promise<VisionExtractionResult> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${apiKey}`

  const body = {
    contents: [
      {
        parts: [
          { text: EXTRACTION_PROMPT },
          { inline_data: { mime_type: mimeType, data: base64 } },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 1024,
      responseMimeType: 'application/json',
    },
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const errorText = await response.text()
    if (response.status === 400 && errorText.includes('API_KEY_INVALID')) {
      throw new Error('Invalid Gemini API key')
    }
    if (response.status === 429) {
      throw new Error('Gemini API rate limit exceeded — try again in a moment')
    }
    throw new Error(`Gemini API error (${response.status}): ${errorText.slice(0, 200)}`)
  }

  const data = await response.json()
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text
  if (!text) {
    throw new Error('No content in Gemini response')
  }

  const parsed = JSON.parse(text) as VisionExtractionResult
  return {
    weapon_name: parsed.weapon_name ?? '',
    mod_names: Array.isArray(parsed.mod_names) ? parsed.mod_names : [],
  }
}
