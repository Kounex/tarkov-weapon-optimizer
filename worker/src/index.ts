/**
 * Cloudflare Worker — proxies screenshot extraction requests to Gemini Flash.
 *
 * The GEMINI_API_KEY is stored as a Worker secret and never exposed to clients.
 * Accepts POST /extract with a JSON body containing { image: base64, mimeType: string }.
 * Returns the Gemini response (weapon_name + mod_names) as JSON.
 */

interface Env {
  GEMINI_API_KEY: string
  ALLOWED_ORIGIN: string
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

function corsHeaders(origin: string, allowedOrigin: string): Record<string, string> {
  const allowed =
    origin === allowedOrigin ||
    origin === 'http://localhost:5173' ||
    origin === 'http://localhost:4173'
  return {
    'Access-Control-Allow-Origin': allowed ? origin : allowedOrigin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const origin = request.headers.get('Origin') ?? ''
    const cors = corsHeaders(origin, env.ALLOWED_ORIGIN)

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors })
    }

    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { ...cors, 'Content-Type': 'application/json' },
      })
    }

    const url = new URL(request.url)
    if (url.pathname !== '/extract') {
      return new Response(JSON.stringify({ error: 'Not found' }), {
        status: 404,
        headers: { ...cors, 'Content-Type': 'application/json' },
      })
    }

    try {
      const body = (await request.json()) as { image?: string; mimeType?: string }

      if (!body.image) {
        return new Response(JSON.stringify({ error: 'Missing "image" (base64) in request body' }), {
          status: 400,
          headers: { ...cors, 'Content-Type': 'application/json' },
        })
      }

      const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${env.GEMINI_API_KEY}`

      const geminiBody = {
        contents: [
          {
            parts: [
              { text: EXTRACTION_PROMPT },
              {
                inline_data: {
                  mime_type: body.mimeType ?? 'image/png',
                  data: body.image,
                },
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.1,
          maxOutputTokens: 8192,
          responseMimeType: 'application/json',
          thinkingConfig: {
            thinkingBudget: 2048,
          },
        },
      }

      const geminiResponse = await fetch(geminiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geminiBody),
      })

      if (!geminiResponse.ok) {
        const errorText = await geminiResponse.text()
        console.error('Gemini API error:', geminiResponse.status, errorText.slice(0, 500))
        return new Response(
          JSON.stringify({ error: `Vision API error (${geminiResponse.status})` }),
          {
            status: 502,
            headers: { ...cors, 'Content-Type': 'application/json' },
          },
        )
      }

      const geminiData = (await geminiResponse.json()) as {
        candidates?: Array<{ content?: { parts?: Array<{ text?: string; thought?: boolean }> } }>
      }

      // With thinking enabled, parts[0] may be a thought — find the first non-thought text part
      const parts = geminiData?.candidates?.[0]?.content?.parts ?? []
      const text = parts.find(p => p.text && !p.thought)?.text ?? parts.find(p => p.text)?.text
      if (!text) {
        return new Response(JSON.stringify({ error: 'No content in vision response' }), {
          status: 502,
          headers: { ...cors, 'Content-Type': 'application/json' },
        })
      }

      const parsed = JSON.parse(text) as { weapon_name?: string; mod_names?: string[] }

      return new Response(
        JSON.stringify({
          weapon_name: parsed.weapon_name ?? '',
          mod_names: Array.isArray(parsed.mod_names) ? parsed.mod_names : [],
        }),
        {
          status: 200,
          headers: { ...cors, 'Content-Type': 'application/json' },
        },
      )
    } catch (err) {
      console.error('Worker error:', err)
      return new Response(
        JSON.stringify({ error: err instanceof Error ? err.message : 'Internal error' }),
        {
          status: 500,
          headers: { ...cors, 'Content-Type': 'application/json' },
        },
      )
    }
  },
}
