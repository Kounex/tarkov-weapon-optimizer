import { useTranslation } from 'react-i18next'
import { Collapse, Checkbox, Slider, Segmented, Space, Divider, Typography, Button, Tooltip, Input } from 'antd'
import { UndoOutlined, CheckCircleFilled, CloseCircleFilled, LinkOutlined } from '@ant-design/icons'
import { DEFAULT_TRADER_LEVELS } from '../../solver/types'
import type { TraderLevels, TarkovTrackerLinkState } from '../../solver/types'

const { Text, Link } = Typography

interface LevelConfigProps {
  fleaAvailable: boolean
  onFleaChange: (v: boolean) => void
  barterAvailable: boolean
  onBarterChange: (v: boolean) => void
  barterExcludeDogtags: boolean
  onBarterExcludeDogsChange: (v: boolean) => void
  excludeScarce: boolean
  onExcludeScarceChange: (v: boolean) => void
  playerLevel: number
  onPlayerLevelChange: (v: number) => void
  traderLevels: TraderLevels
  onTraderLevelsChange: (v: TraderLevels) => void
  tarkovTracker: TarkovTrackerLinkState
  onTarkovTrackerTokenChange: (v: string) => void
  onTarkovTrackerLink: () => void
  onTarkovTrackerUnlink: () => void
}

function TarkovTrackerSection({
  tarkovTracker, onTarkovTrackerTokenChange, onTarkovTrackerLink, onTarkovTrackerUnlink,
}: Pick<LevelConfigProps, 'tarkovTracker' | 'onTarkovTrackerTokenChange' | 'onTarkovTrackerLink' | 'onTarkovTrackerUnlink'>) {
  const { t } = useTranslation()
  const linked = tarkovTracker.status === 'linked'
  return (
    <Space direction="vertical" style={{ width: '100%' }} size={8}>
      <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.tarkovtracker_desc')}</Text>
      <Space.Compact style={{ width: '100%' }}>
        <Input.Password
          size="small"
          placeholder={t('sidebar.tarkovtracker_token_placeholder')}
          value={tarkovTracker.token}
          onChange={(e) => onTarkovTrackerTokenChange(e.target.value)}
          disabled={linked}
        />
        {linked ? (
          <Button size="small" danger onClick={onTarkovTrackerUnlink}>{t('sidebar.tarkovtracker_unlink')}</Button>
        ) : (
          <Button
            size="small"
            type="primary"
            icon={<LinkOutlined />}
            loading={tarkovTracker.status === 'checking'}
            disabled={!tarkovTracker.token.trim()}
            onClick={onTarkovTrackerLink}
          >
            {t('sidebar.tarkovtracker_link')}
          </Button>
        )}
      </Space.Compact>
      {tarkovTracker.status === 'linked' && (
        <Text type="success" style={{ fontSize: 12 }}>
          <CheckCircleFilled /> {t('sidebar.tarkovtracker_linked_as', { name: tarkovTracker.displayName || '?' })}
        </Text>
      )}
      {tarkovTracker.status === 'error' && (
        <Text type="danger" style={{ fontSize: 12 }}>
          <CloseCircleFilled /> {tarkovTracker.error || t('sidebar.tarkovtracker_error')}
        </Text>
      )}
      <Text type="secondary" style={{ fontSize: 11 }}>
        {t('sidebar.tarkovtracker_hint')}{' '}
        <Link href="https://tarkovtracker.org/settings" target="_blank" rel="noreferrer">tarkovtracker.org</Link>
      </Text>
    </Space>
  )
}

export function LevelConfig({
  fleaAvailable,
  onFleaChange,
  barterAvailable,
  onBarterChange,
  barterExcludeDogtags,
  onBarterExcludeDogsChange,
  excludeScarce,
  onExcludeScarceChange,
  playerLevel,
  onPlayerLevelChange,
  traderLevels,
  onTraderLevelsChange,
  tarkovTracker,
  onTarkovTrackerTokenChange,
  onTarkovTrackerLink,
  onTarkovTrackerUnlink,
}: LevelConfigProps) {
  const { t } = useTranslation()
  return (
    <Collapse size="small" items={[
      {
        key: 'market',
        label: <span style={{ userSelect: 'none' }}>{t('sidebar.player_trader_access')}</span>,
        extra: <Tooltip title={t('ui.reset')}><Button type="text" size="small" icon={<UndoOutlined />} onClick={(e) => { e.stopPropagation(); onFleaChange(true); onBarterChange(false); onBarterExcludeDogsChange(true); onExcludeScarceChange(false); onPlayerLevelChange(60); onTraderLevelsChange({ ...DEFAULT_TRADER_LEVELS }) }} /></Tooltip>,
        children: (
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.flea_market_access')}</Text>
              <Segmented size="small" value={fleaAvailable ? 'on' : 'off'} onChange={v => onFleaChange(v === 'on')} options={[{ label: t('ui.on'), value: 'on' }, { label: t('ui.off'), value: 'off' }]} />
            </div>
            {fleaAvailable && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Tooltip title={t('sidebar.exclude_scarce_tooltip')}>
                  <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.exclude_scarce')}</Text>
                </Tooltip>
                <Segmented size="small" value={excludeScarce ? 'on' : 'off'} onChange={v => onExcludeScarceChange(v === 'on')} options={[{ label: t('ui.on'), value: 'on' }, { label: t('ui.off'), value: 'off' }]} />
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.barter_available')}</Text>
              <Segmented size="small" value={barterAvailable ? 'on' : 'off'} onChange={v => onBarterChange(v === 'on')} options={[{ label: t('ui.on'), value: 'on' }, { label: t('ui.off'), value: 'off' }]} />
            </div>
            {barterAvailable && (
              <Checkbox checked={barterExcludeDogtags} onChange={(e) => onBarterExcludeDogsChange(e.target.checked)}>
                <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.barter_exclude_dogtags')}</Text>
              </Checkbox>
            )}
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>{t('sidebar.player_level')}: {playerLevel}</Text>
              <Slider value={playerLevel} onChange={onPlayerLevelChange} min={1} max={79} />
            </div>
            <Divider style={{ margin: '8px 0' }} />
            {(Object.keys(traderLevels) as Array<keyof TraderLevels>).map(trader => (
              <div key={trader} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <Text type="secondary" style={{ fontSize: 12, minWidth: 70 }}>{t(`trader.${trader}`)}</Text>
                <Segmented size="small" value={traderLevels[trader]} onChange={(v) => onTraderLevelsChange({ ...traderLevels, [trader]: v as number })} options={[{ label: t('ui.ll_level', { level: 1 }), value: 1 }, { label: t('ui.ll_level', { level: 2 }), value: 2 }, { label: t('ui.ll_level', { level: 3 }), value: 3 }, { label: t('ui.ll_level', { level: 4 }), value: 4 }]} />
              </div>
            ))}
          </Space>
        ),
      },
      {
        key: 'tarkovtracker',
        label: <span style={{ userSelect: 'none' }}>{t('sidebar.tarkovtracker_title')}</span>,
        children: (
          <TarkovTrackerSection
            tarkovTracker={tarkovTracker}
            onTarkovTrackerTokenChange={onTarkovTrackerTokenChange}
            onTarkovTrackerLink={onTarkovTrackerLink}
            onTarkovTrackerUnlink={onTarkovTrackerUnlink}
          />
        ),
      },
    ]} />
  )
}
