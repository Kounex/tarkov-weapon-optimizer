import { useTranslation } from 'react-i18next'
import {
  Card, Space, Typography, InputNumber, Checkbox, Tooltip, Tag,
} from 'antd'
import { DollarOutlined } from '@ant-design/icons'
import { WeightAdjuster } from '../common/WeightAdjuster'
import { LevelConfig } from '../common/LevelConfig'
import type { TraderLevels } from '../../solver/types'

const { Paragraph } = Typography

interface ImportSidebarProps {
  ergoWeight: number
  recoilWeight: number
  priceWeight: number
  onWeightChange: (ergo: number, recoil: number, price: number) => void
  useUpgradeBudget: boolean
  onUseUpgradeBudgetChange: (v: boolean) => void
  upgradeBudget: number
  onUpgradeBudgetChange: (v: number) => void
  minErgo: number
  onMinErgoChange: (v: number) => void
  fleaAvailable: boolean
  onFleaChange: (v: boolean) => void
  barterAvailable: boolean
  onBarterChange: (v: boolean) => void
  barterExcludeDogtags: boolean
  onBarterExcludeDogsChange: (v: boolean) => void
  playerLevel: number
  onPlayerLevelChange: (v: number) => void
  traderLevels: TraderLevels
  onTraderLevelsChange: (v: TraderLevels) => void
  lockedModsCost: number
}

export function ImportSidebar({
  ergoWeight,
  recoilWeight,
  priceWeight,
  onWeightChange,
  useUpgradeBudget,
  onUseUpgradeBudgetChange,
  upgradeBudget,
  onUpgradeBudgetChange,
  minErgo,
  onMinErgoChange,
  fleaAvailable,
  onFleaChange,
  barterAvailable,
  onBarterChange,
  barterExcludeDogtags,
  onBarterExcludeDogsChange,
  playerLevel,
  onPlayerLevelChange,
  traderLevels,
  onTraderLevelsChange,
  lockedModsCost,
}: ImportSidebarProps) {
  const { t } = useTranslation()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <WeightAdjuster
        ergoWeight={ergoWeight}
        recoilWeight={recoilWeight}
        priceWeight={priceWeight}
        onWeightChange={onWeightChange}
        useBudget={false}
        onUseBudgetChange={() => {}}
        maxPrice={0}
        onMaxPriceChange={() => {}}
        minErgo={minErgo}
        onMinErgoChange={onMinErgoChange}
        useMinMag={false}
        onUseMinMagChange={() => {}}
        minMagCapacity={0}
        onMinMagCapacityChange={() => {}}
        availableMagCapacities={[]}
        useMOA={false}
        onUseMOAChange={() => {}}
        maxMOA={0}
        onMaxMOAChange={() => {}}
        moaRange={{ base: 0, min: 0, max: 0 }}
        useExactMOAFloor={false}
        onUseExactMOAFloorChange={() => {}}
        computingMOAFloor={false}
      />

      <Card size="small" title={<Space><DollarOutlined />{t('import.upgrade_budget_title')}</Space>}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>
            {t('import.upgrade_budget_help')}
          </Paragraph>
          <Space>
            <Checkbox
              checked={useUpgradeBudget}
              onChange={e => onUseUpgradeBudgetChange(e.target.checked)}
            >
              {t('import.upgrade_budget_enable')}
            </Checkbox>
          </Space>
          {useUpgradeBudget && (
            <Space>
              <InputNumber
                value={upgradeBudget}
                onChange={v => onUpgradeBudgetChange(v ?? 0)}
                min={0}
                step={10000}
                formatter={v => `₽ ${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                parser={v => Number((v ?? '').replace(/[₽\s,]/g, '')) || 0}
                style={{ width: 180 }}
              />
              {lockedModsCost > 0 && (
                <Tooltip title={t('import.locked_mods_cost_tooltip')}>
                  <Tag color="blue">
                    {t('import.locked_mods_cost', { cost: lockedModsCost.toLocaleString() })}
                  </Tag>
                </Tooltip>
              )}
            </Space>
          )}
        </Space>
      </Card>

      <LevelConfig
        fleaAvailable={fleaAvailable}
        onFleaChange={onFleaChange}
        barterAvailable={barterAvailable}
        onBarterChange={onBarterChange}
        barterExcludeDogtags={barterExcludeDogtags}
        onBarterExcludeDogsChange={onBarterExcludeDogsChange}
        playerLevel={playerLevel}
        onPlayerLevelChange={onPlayerLevelChange}
        traderLevels={traderLevels}
        onTraderLevelsChange={onTraderLevelsChange}
      />
    </div>
  )
}
