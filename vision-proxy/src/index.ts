/**
 * Vision proxy — lightweight Hono server that proxies screenshot extraction
 * requests to Gemini Pro.  Keeps the API key server-side.
 *
 * Environment variables:
 *   GEMINI_API_KEY   — required
 *   ALLOWED_ORIGIN   — optional, defaults to "*"
 *   PORT             — optional, defaults to 3001
 */

import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serve } from '@hono/node-server'

const app = new Hono()

const GEMINI_API_KEY = process.env.GEMINI_API_KEY ?? ''
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN ?? '*'
const PORT = Number(process.env.PORT ?? 3001)

if (!GEMINI_API_KEY) {
  console.error('GEMINI_API_KEY is required')
  process.exit(1)
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

app.use('/extract', cors({
  origin: ALLOWED_ORIGIN === '*' ? '*' : [ALLOWED_ORIGIN, 'http://localhost:5173', 'http://localhost:4173'],
  allowMethods: ['POST', 'OPTIONS'],
  allowHeaders: ['Content-Type'],
  maxAge: 86400,
}))

app.get('/health', (c) => c.json({ status: 'ok' }))

app.post('/extract', async (c) => {
  const body = await c.req.json<{ image?: string; mimeType?: string }>()

  if (!body.image) {
    return c.json({ error: 'Missing "image" (base64) in request body' }, 400)
  }

  const geminiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GEMINI_API_KEY}`

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

  try {
    const geminiResponse = await fetch(geminiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(geminiBody),
    })

    if (!geminiResponse.ok) {
      const errorText = await geminiResponse.text()
      console.error('Gemini API error:', geminiResponse.status, errorText.slice(0, 500))
      return c.json({ error: `Vision API error (${geminiResponse.status})` }, 502)
    }

    const geminiData = (await geminiResponse.json()) as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string; thought?: boolean }> } }>
    }

    const parts = geminiData?.candidates?.[0]?.content?.parts ?? []
    const text = parts.find(p => p.text && !p.thought)?.text ?? parts.find(p => p.text)?.text

    if (!text) {
      return c.json({ error: 'No content in vision response' }, 502)
    }

    const parsed = JSON.parse(text) as { weapon_name?: string; mod_names?: string[] }

    return c.json({
      weapon_name: parsed.weapon_name ?? '',
      mod_names: Array.isArray(parsed.mod_names) ? parsed.mod_names : [],
    })
  } catch (err) {
    console.error('Proxy error:', err)
    return c.json(
      { error: err instanceof Error ? err.message : 'Internal error' },
      500,
    )
  }
})

console.log(`Vision proxy listening on :${PORT}`)
serve({ fetch: app.fetch, port: PORT })
