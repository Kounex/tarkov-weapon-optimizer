import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Card, Select, Space, Row, Col, Typography, Tag } from 'antd'
import type { Gun, WeaponPresetOption } from '../../api/client'
import { TraderIcon } from '../ItemRow'

const { Text } = Typography

interface WeaponSelectorProps {
  guns: Gun[]
  selectedGunId: string
  onGunChange: (id: string) => void
  selectedCategory: string
  onCategoryChange: (category: string) => void
  selectedCaliber: string
  onCaliberChange: (caliber: string) => void
  categories: string[]
  calibers: string[]
  filteredGuns: Gun[]
  // Optional forced-base preset selector (rendered only when onPresetChange is provided)
  presets?: WeaponPresetOption[]
  nakedBase?: { price: number; source?: string | null; available: boolean } | null
  selectedPresetId?: string
  onPresetChange?: (id: string | undefined) => void
  loadingPresets?: boolean
  /** id → name/image of every preset seen for this weapon (labels for options that became unavailable at current settings) */
  presetNameLookup?: Record<string, { name: string; image?: string | null }>
  /** Same list as ModFilter's excludedModIds — a base (preset id, or a weapon's own id for its naked receiver) can be banned too. */
  excludedBaseIds?: string[]
  onToggleExcludeBase?: (id: string) => void
}

export function WeaponSelector({
  guns,
  selectedGunId,
  onGunChange,
  selectedCategory,
  onCategoryChange,
  selectedCaliber,
  onCaliberChange,
  categories,
  calibers,
  filteredGuns,
  presets,
  nakedBase,
  selectedPresetId,
  onPresetChange,
  loadingPresets,
  presetNameLookup,
  excludedBaseIds,
  onToggleExcludeBase,
}: WeaponSelectorProps) {
  const { t } = useTranslation()
  const [searchValue, setSearchValue] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)

  // Banned bases resolve their display name either from the current weapon
  // (naked receiver — banned under the weapon's own id) or from
  // presetNameLookup (a preset can only be banned from a build result after
  // already being fetched/shown once, so its name is always cached by then).
  const bannedBaseEntries = (excludedBaseIds ?? [])
    .map(id => {
      if (id === selectedGunId) {
        const gunName = guns.find(g => g.id === id)?.name
        return gunName ? { id, name: `${t('ui.preset_base_stock')} — ${gunName}` } : null
      }
      const known = presetNameLookup?.[id]
      return known ? { id, name: known.name } : null
    })
    .filter((e): e is { id: string; name: string } => e !== null)

  // The selected preset may be unavailable at the current trader/flea settings
  // (e.g. user tightened trader levels after picking it) — keep it visible in
  // the dropdown, marked, so the solver's auto-fallback warning makes sense.
  const selectedUnavailable =
    selectedPresetId && selectedPresetId !== 'naked' && !(presets ?? []).some(p => p.id === selectedPresetId)
      ? presetNameLookup?.[selectedPresetId]
      : undefined

  const presetOptions = [
    { value: 'auto', label: t('ui.preset_base_auto') },
    {
      value: 'naked',
      label: nakedBase?.available
        ? `${t('ui.preset_base_stock')} — ₽${nakedBase.price.toLocaleString()}`
        : t('ui.preset_base_stock'),
    },
    ...(presets ?? []).map(p => ({
      value: p.id,
      label: `${p.name} — ₽${p.price.toLocaleString()}`,
    })),
    ...(selectedPresetId && selectedUnavailable
      ? [{ value: selectedPresetId, label: `${selectedUnavailable.name} (${t('ui.preset_base_unavailable')})` }]
      : []),
  ]

  return (
    <Card title={<span style={{ userSelect: 'none' }}>{t('sidebar.select_weapon')}</span>} size="small">
      <Space direction="vertical" style={{ width: '100%' }}>
        <Row gutter={8}>
          <Col span={12}>
            <Select
              style={{ width: '100%' }}
              value={selectedCategory === 'All' ? undefined : selectedCategory}
              onChange={(v) => onCategoryChange(v || 'All')}
              placeholder={t('ui.weapon_category')}
              allowClear
              options={categories.filter(c => c !== 'All').map(c => ({ value: c, label: c }))}
            />
          </Col>
          <Col span={12}>
            <Select
              style={{ width: '100%' }}
              value={selectedCaliber === 'All' ? undefined : selectedCaliber}
              onChange={(v) => onCaliberChange(v || 'All')}
              placeholder={t('ui.caliber_type')}
              allowClear
              options={calibers.filter(c => c !== 'All').map(c => ({ value: c, label: c }))}
            />
          </Col>
        </Row>
        <Select
          showSearch
          style={{ width: '100%' }}
          value={selectedGunId}
          searchValue={searchValue}
          onSearch={setSearchValue}
          onDropdownVisibleChange={setDropdownOpen}
          onChange={(v) => { onGunChange(v); setSearchValue('') }}
          onKeyDown={(e) => { if (e.key === ' ' && dropdownOpen) setSearchValue(prev => prev + ' ') }}
          labelRender={(item) => (
            dropdownOpen && searchValue
              ? <span style={{ visibility: 'hidden' }}>{item.label}</span>
              : <span>{item.label}</span>
          )}
          filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
          options={filteredGuns.map(g => ({ value: g.id, label: g.name }))}
          optionRender={(option) => {
            const gun = filteredGuns.find(g => g.id === option.value)
            return (
              <Space>
                {gun?.image && <img src={gun.image} alt="" style={{ width: 48, height: 32, objectFit: 'contain' }} />}
                <span>{option.label}</span>
              </Space>
            )
          }}
        />
        {onPresetChange && selectedGunId && (
          <div>
            <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>{t('ui.preset_base_label')}</Text>
            <Select
              showSearch
              style={{ width: '100%' }}
              popupMatchSelectWidth={false}
              loading={loadingPresets}
              value={selectedPresetId ?? 'auto'}
              onChange={(v) => onPresetChange(v === 'auto' ? undefined : v)}
              filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
              options={presetOptions}
              optionRender={(option) => {
                if (option.value === 'auto' || option.value === 'naked') {
                  const showNakedPrice = option.value === 'naked' && nakedBase?.available
                  return (
                    <Space>
                      <span>{option.value === 'naked' ? t('ui.preset_base_stock') : option.label}</span>
                      {option.value === 'naked' && <TraderIcon source={nakedBase?.source ?? undefined} unknownLabel={t('ui.unknown')} compact />}
                      {showNakedPrice && <Text type="secondary" style={{ fontSize: 12 }}>₽{nakedBase!.price.toLocaleString()}</Text>}
                    </Space>
                  )
                }
                const preset = (presets ?? []).find(p => p.id === option.value)
                if (!preset) {
                  // Selected but unavailable at current settings
                  return <Text type="warning">{option.label}</Text>
                }
                return (
                  <Space>
                    {preset.image && <img src={preset.image} alt="" style={{ width: 64, height: 32, objectFit: 'contain' }} />}
                    <span>{preset.name}</span>
                    <TraderIcon source={preset.source ?? undefined} unknownLabel={t('ui.unknown')} compact />
                    <Text type="secondary" style={{ fontSize: 12 }}>₽{preset.price.toLocaleString()}</Text>
                  </Space>
                )
              }}
            />
            {bannedBaseEntries.length > 0 && (
              <Space wrap style={{ marginTop: 8 }}>
                {bannedBaseEntries.map(({ id, name }) => (
                  <Tag key={id} color="error" closable onClose={() => onToggleExcludeBase?.(id)}>
                    {name}
                  </Tag>
                ))}
              </Space>
            )}
          </div>
        )}
      </Space>
    </Card>
  )
}
