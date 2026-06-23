import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Card, Button, Upload, Input, Space, Typography, Alert, List, Switch,
  Tag, Tooltip, Collapse, Spin, theme, Image,
} from 'antd'
import {
  CameraOutlined, KeyOutlined, LockOutlined, UnlockOutlined,
  CheckCircleFilled, QuestionCircleFilled, WarningFilled,
  DeleteOutlined, EyeOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import type { Gun, ModInfo, OptimizeResponse } from '../../api/client'
import { extractFromScreenshot, getStoredApiKey, setStoredApiKey, getProxyUrl } from '../../api/visionService'
import { matchWeapon, matchMods, type MatchedMod, type MatchedWeapon } from '../../api/screenshotMatcher'
import { WeaponSelector } from '../common/WeaponSelector'
import { StatsCards } from '../common/StatsCards'
import { BuildManifest } from '../common/BuildManifest'
import { UsingPresetCard } from '../common/UsingPresetCard'

const { Text, Paragraph } = Typography

interface ImportPanelProps {
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
  availableMods: ModInfo[]
  loadingMods: boolean
  onLockedIdsChange: (ids: string[]) => void
  result: OptimizeResponse | null
  optimizing: boolean
  onOptimize: () => void
  onCopy: () => void
  viewMode: 'detailed' | 'compact' | 'table'
  onViewModeChange: (v: 'detailed' | 'compact' | 'table') => void
  lockedIds: string[]
  onToggleLock?: (id: string) => void
}

export function ImportPanel({
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
  availableMods,
  loadingMods,
  onLockedIdsChange,
  result,
  optimizing,
  onOptimize,
  onCopy,
  viewMode,
  onViewModeChange,
  lockedIds,
  onToggleLock,
}: ImportPanelProps) {
  const { t } = useTranslation()
  const { token } = theme.useToken()
  const [apiKey, setApiKey] = useState(getStoredApiKey)
  const [showKey, setShowKey] = useState(false)
  const hasProxy = !!getProxyUrl()
  const [screenshot, setScreenshot] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [extracting, setExtracting] = useState(false)
  const [extractError, setExtractError] = useState<string | null>(null)
  const [matchedWeapon, setMatchedWeapon] = useState<MatchedWeapon | null>(null)
  const [matchedMods, setMatchedMods] = useState<MatchedMod[]>([])
  const [extracted, setExtracted] = useState(false)
  const [extractedModNames, setExtractedModNames] = useState<string[]>([])
  const [modsMatched, setModsMatched] = useState(false)
  const handleApiKeyChange = (value: string) => {
    setApiKey(value)
    setStoredApiKey(value)
  }

  useEffect(() => {
    if (modsMatched) {
      setModsMatched(false)
      setMatchedMods([])
      onLockedIdsChange([])
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedGunId])

  const visionConfigured = hasProxy || !!apiKey

  const handleFileSelect = (file: File) => {
    setScreenshot(file)
    setExtracted(false)
    setMatchedMods([])
    setMatchedWeapon(null)
    setExtractError(null)

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleExtract = async () => {
    if (!screenshot || !visionConfigured) return

    setExtracting(true)
    setExtractError(null)

    try {
      const result = await extractFromScreenshot(screenshot)

      const weaponMatch = matchWeapon(result.weapon_name, guns)
      setMatchedWeapon(weaponMatch)

      if (weaponMatch) {
        onGunChange(weaponMatch.gun.id)
      }

      setExtractedModNames(result.mod_names)
      setExtracted(true)
      setModsMatched(false)
      setMatchedMods([])
    } catch (err) {
      setExtractError(err instanceof Error ? err.message : String(err))
    } finally {
      setExtracting(false)
    }
  }

  const handleMatchMods = () => {
    if (extractedModNames.length === 0 || availableMods.length === 0) return
    const modMatches = matchMods(extractedModNames, availableMods)
    setMatchedMods(modMatches)
    setModsMatched(true)

    const locked = modMatches
      .filter(m => m.locked && m.mod)
      .map(m => m.mod!.id)
    onLockedIdsChange(locked)
  }

  const toggleModLock = (index: number) => {
    setMatchedMods(prev => {
      const next = [...prev]
      next[index] = { ...next[index], locked: !next[index].locked }

      const locked = next
        .filter(m => m.locked && m.mod)
        .map(m => m.mod!.id)
      onLockedIdsChange(locked)

      return next
    })
  }

  const handleClear = () => {
    setScreenshot(null)
    setPreviewUrl(null)
    setExtracted(false)
    setMatchedMods([])
    setMatchedWeapon(null)
    setExtractError(null)
    setExtractedModNames([])
    setModsMatched(false)
    onLockedIdsChange([])
  }

  const confidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <CheckCircleFilled style={{ color: token.colorSuccess }} />
    if (confidence >= 0.5) return <QuestionCircleFilled style={{ color: token.colorWarning }} />
    return <WarningFilled style={{ color: token.colorError }} />
  }

  const confidenceTag = (confidence: number) => {
    if (confidence >= 0.8) return <Tag color="success">{Math.round(confidence * 100)}%</Tag>
    if (confidence >= 0.5) return <Tag color="warning">{Math.round(confidence * 100)}%</Tag>
    return <Tag color="error">{Math.round(confidence * 100)}%</Tag>
  }

  const precisionResolvedLabel = (mode: 'fast' | 'precise') =>
    mode === 'precise' ? t('sidebar.precise') : t('sidebar.fast')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Step 1: Vision Service Config */}
      {!hasProxy && (
        <Collapse
          size="small"
          items={[{
            key: 'apikey',
            label: (
              <Space>
                <KeyOutlined />
                <span>{t('import.api_key_title')}</span>
                {apiKey ? <Tag color="success">{t('import.api_key_set')}</Tag> : <Tag color="error">{t('import.api_key_missing')}</Tag>}
              </Space>
            ),
            children: (
              <Space direction="vertical" style={{ width: '100%' }}>
                <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>
                  {t('import.api_key_help')}
                </Paragraph>
                <Input.Password
                  placeholder="AIza..."
                  value={apiKey}
                  onChange={e => handleApiKeyChange(e.target.value)}
                  visibilityToggle={{ visible: showKey, onVisibleChange: setShowKey }}
                  addonBefore={<KeyOutlined />}
                />
              </Space>
            ),
          }]}
        />
      )}

      {/* Step 2: Screenshot Upload */}
      <Card size="small" title={<Space><CameraOutlined />{t('import.screenshot_title')}</Space>}>
        {!screenshot ? (
          <Upload.Dragger
            accept="image/*"
            showUploadList={false}
            beforeUpload={(file) => {
              handleFileSelect(file)
              return false
            }}
            style={{ padding: '20px 0' }}
          >
            <p style={{ marginBottom: 8, fontSize: 14 }}>
              <CameraOutlined style={{ fontSize: 24, color: token.colorTextSecondary }} />
            </p>
            <Paragraph type="secondary" style={{ margin: 0 }}>
              {t('import.upload_hint')}
            </Paragraph>
          </Upload.Dragger>
        ) : (
          <Space direction="vertical" style={{ width: '100%' }}>
            {previewUrl && (
              <Image
                src={previewUrl}
                alt="Screenshot preview"
                style={{ maxHeight: 200, objectFit: 'contain', borderRadius: token.borderRadius }}
                preview={{ mask: <Space><EyeOutlined />{t('import.preview')}</Space> }}
              />
            )}
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>{screenshot.name}</Text>
              <Button size="small" icon={<DeleteOutlined />} onClick={handleClear} danger>
                {t('import.clear')}
              </Button>
            </Space>
            <Button
              type="primary"
              icon={<CameraOutlined />}
              onClick={handleExtract}
              loading={extracting}
              disabled={!visionConfigured}
              block
            >
              {extracting ? t('import.extracting') : t('import.extract_btn')}
            </Button>
          </Space>
        )}
      </Card>

      {extractError && (
        <Alert type="error" message={extractError} closable onClose={() => setExtractError(null)} />
      )}

      {/* Step 3: Weapon Match + Selection */}
      {extracted && (
        <>
          <Card
            size="small"
            title={t('import.detected_weapon')}
            extra={matchedWeapon ? confidenceTag(matchedWeapon.confidence) : <Tag color="error">{t('import.not_found')}</Tag>}
          >
            {matchedWeapon ? (
              <Text>{matchedWeapon.gun.name}</Text>
            ) : (
              <Text type="secondary">{t('import.weapon_not_matched')}</Text>
            )}
          </Card>

          <WeaponSelector
            guns={guns}
            selectedGunId={selectedGunId}
            onGunChange={onGunChange}
            selectedCategory={selectedCategory}
            onCategoryChange={onCategoryChange}
            selectedCaliber={selectedCaliber}
            onCaliberChange={onCaliberChange}
            categories={categories}
            calibers={calibers}
            filteredGuns={filteredGuns}
          />

          {/* Step 4: Match mods button */}
          {extractedModNames.length > 0 && !modsMatched && (
            <Button
              type="primary"
              onClick={handleMatchMods}
              loading={loadingMods}
              disabled={!selectedGunId || loadingMods}
              block
              size="large"
            >
              {loadingMods ? t('import.loading_mods') : t('import.match_mods_btn')}
            </Button>
          )}
        </>
      )}

      {/* Step 5: Matched Mods with Lock/Unlock */}
      {extracted && modsMatched && matchedMods.length > 0 && (
        <Card
          size="small"
          title={
            <Space>
              <span>{t('import.detected_mods')}</span>
              <Tag>{matchedMods.filter(m => m.locked).length} / {matchedMods.length} {t('import.locked')}</Tag>
            </Space>
          }
        >
          {loadingMods ? (
            <div style={{ textAlign: 'center', padding: 20 }}><Spin /></div>
          ) : (
            <List
              size="small"
              dataSource={matchedMods}
              renderItem={(match, index) => (
                <List.Item
                  style={{
                    padding: '6px 0',
                    opacity: match.mod ? 1 : 0.5,
                    background: match.locked ? `${token.colorPrimaryBg}40` : undefined,
                    borderRadius: token.borderRadius,
                    paddingLeft: 8,
                    paddingRight: 8,
                  }}
                  actions={[
                    match.mod && (
                      <Tooltip title={match.locked ? t('import.unlock_tooltip') : t('import.lock_tooltip')} key="lock">
                        <Switch
                          size="small"
                          checked={match.locked}
                          onChange={() => toggleModLock(index)}
                          checkedChildren={<LockOutlined />}
                          unCheckedChildren={<UnlockOutlined />}
                        />
                      </Tooltip>
                    ),
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={
                      match.mod?.icon ? (
                        <img src={match.mod.icon} alt="" style={{ width: 48, height: 48, objectFit: 'contain' }} />
                      ) : (
                        confidenceIcon(match.confidence)
                      )
                    }
                    title={
                      <Space size={4}>
                        {match.mod?.icon && confidenceIcon(match.confidence)}
                        <Text style={{ fontSize: 13 }}>
                          {match.mod?.name ?? match.extractedName}
                        </Text>
                      </Space>
                    }
                    description={
                      match.mod ? (
                        <Space size={4}>
                          <Text type="secondary" style={{ fontSize: 11 }}>
                            {t('import.matched_from')}: "{match.extractedName}"
                          </Text>
                          {confidenceTag(match.confidence)}
                        </Space>
                      ) : (
                        <Text type="danger" style={{ fontSize: 11 }}>
                          {t('import.no_match')}: "{match.extractedName}"
                        </Text>
                      )
                    }
                  />
                </List.Item>
              )}
            />
          )}
        </Card>
      )}

      {/* Step 6: Optimize */}
      {modsMatched && (
        <Button
          type="primary"
          icon={<ThunderboltOutlined />}
          loading={optimizing}
          disabled={!selectedGunId}
          onClick={onOptimize}
          block
          size="large"
        >
          {optimizing ? t('import.extracting') : t('import.optimize_btn')}
        </Button>
      )}

      {/* Results */}
      {result && (
        <>
          <Alert
            type={result.status === 'optimal' ? 'success' : result.status === 'infeasible' ? 'error' : 'warning'}
            message={
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, width: '100%' }}>
                <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                  <Text>{t('results.optimization_status')}: {t(`results.status_${result.status}`, { defaultValue: result.status })}</Text>
                  {lockedIds.length > 0 && (
                    <Tag color="blue">{t('import.kept_parts', { count: lockedIds.length })}</Tag>
                  )}
                  {typeof result.solve_time_ms === 'number' && <Tag color="blue" style={{ margin: 0 }}>{result.solve_time_ms.toFixed(0)} ms</Tag>}
                  {result.precision_request === 'auto' && result.precision_resolved && (
                    <Tag color="processing" style={{ margin: 0 }} title={t('sidebar.solver_precision_tooltip')}>
                      {t('results.precision_auto_ran', {
                        mode: precisionResolvedLabel(result.precision_resolved),
                      })}
                    </Tag>
                  )}
                </div>
              </div>
            }
            style={{ padding: '12px 16px' }}
          />
          {result.status === 'optimal' && result.final_stats && (
            <>
              <StatsCards
                ergonomics={result.final_stats.ergonomics}
                recoilVertical={result.final_stats.recoil_vertical}
                recoilHorizontal={result.final_stats.recoil_horizontal}
                weight={result.final_stats.total_weight}
                price={result.final_stats.total_price}
                moa={result.final_stats.moa}
              />
              {result.selected_preset && <UsingPresetCard preset={result.selected_preset} />}
              <BuildManifest
                result={result}
                viewMode={viewMode}
                onViewModeChange={onViewModeChange}
                onCopy={onCopy}
                weaponId={selectedGunId}
                lockedIds={lockedIds}
                onToggleLock={onToggleLock}
              />
            </>
          )}
          {result.status === 'infeasible' && (
            <Alert
              type="error"
              message={t('toast.optimize_infeasible')}
              description={result.reason}
              showIcon
            />
          )}
        </>
      )}
    </div>
  )
}
