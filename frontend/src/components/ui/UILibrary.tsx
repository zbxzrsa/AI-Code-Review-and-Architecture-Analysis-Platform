/**
 * 通用 UI 组件库
 *
 * 包含：
 * - StatCard: 统计卡片
 * - TrendCard: 趋势卡片 (含 sparkline)
 * - SeverityTag: 严重级别标签
 * - StatusBadge: 状态徽章
 * - EmptyState: 空状态
 * - ErrorBoundary: 错误边界
 * - LoadingSkeleton: 加载骨架屏
 * - SSEStatusBar: SSE 状态条
 */

import React, { ReactNode } from 'react';
import { Card, Tag, Badge, Spin, Skeleton, Button } from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import styled from 'styled-components';
import { getSeverityColor, spacing, fontSize, borderRadius, shadows } from '../../styles/tokens';

// ============ StatCard 样式 ============
const StatCardWrapper = styled(Card)`
  border-radius: ${borderRadius.sm}px;
  box-shadow: ${shadows.xs};
  transition: all 0.3s ease;

  &:hover {
    box-shadow: ${shadows.md};
    transform: translateY(-2px);
  }

  .stat-value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: ${fontSize.sm}px;
    color: var(--text-secondary);
    margin-top: ${spacing.sm}px;
  }

  .stat-trend {
    font-size: 12px;
    margin-top: ${spacing.xs}px;
    display: flex;
    align-items: center;
    gap: 4px;
  }
`;

// ============ EmptyState 样式 ============
const EmptyStateWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${spacing.xl}px;
  text-align: center;
  min-height: 300px;
  background: var(--bg-secondary);
  border-radius: ${borderRadius.sm}px;

  .empty-icon {
    font-size: 48px;
    color: var(--text-tertiary);
    margin-bottom: ${spacing.md}px;
  }

  .empty-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: ${spacing.sm}px;
  }

  .empty-description {
    font-size: ${fontSize.sm}px;
    color: var(--text-secondary);
    max-width: 400px;
    margin-bottom: ${spacing.md}px;
  }
`;

// ============ SSEStatusBar 样式 ============
const SSEStatusBarWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: ${spacing.md}px;
  padding: ${spacing.md}px ${spacing.lg}px;
  background: linear-gradient(90deg, #2e90fa 0%, #1a73e8 100%);
  color: white;
  border-radius: ${borderRadius.sm}px;
  font-size: ${fontSize.sm}px;
  box-shadow: ${shadows.md};

  .task-info {
    flex: 1;
    display: flex;
    align-items: center;
    gap: ${spacing.md}px;
  }

  .metrics {
    display: flex;
    gap: ${spacing.lg}px;
    align-items: center;

    .metric {
      display: flex;
      flex-direction: column;
      align-items: center;

      .label {
        font-size: 11px;
        opacity: 0.9;
      }

      .value {
        font-size: 14px;
        font-weight: 600;
      }
    }
  }
`;

// ============ StatCard 组件 ============
export interface StatCardProps {
  title: string;
  value: number | string;
  prefix?: ReactNode;
  suffix?: ReactNode;
  trend?: { direction: 'up' | 'down'; value: number };
  onClick?: () => void;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  prefix,
  suffix,
  trend,
  onClick,
}) => {
  return (
    <StatCardWrapper
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      <div className="stat-value">
        {prefix}
        {value}
        {suffix}
      </div>
      <div className="stat-label">{title}</div>
      {trend && (
        <div
          className="stat-trend"
          style={{
            color:
              trend.direction === 'up' ? '#12B76A' : '#F04438',
          }}
        >
          {trend.direction === 'up' ? (
            <ArrowUpOutlined />
          ) : (
            <ArrowDownOutlined />
          )}
          <span>{trend.value}%</span>
        </div>
      )}
    </StatCardWrapper>
  );
};

// ============ TrendCard 组件 ============
export interface TrendCardProps {
  title: string;
  data: Array<{ time: string; value: number }>;
  xField?: string;
  yField?: string;
  height?: number;
}

export const TrendCard: React.FC<TrendCardProps> = ({
  title,
  data,
  xField = 'time',
  yField = 'value',
  height = 300,
}) => {
  // 简化版本：返回卡片占位符（完整版需要 @ant-design/plots）
  return (
    <Card
      title={title}
      style={{
        borderRadius: borderRadius.sm,
        boxShadow: shadows.xs,
      }}
    >
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-tertiary)',
        }}
      >
        趋势图表 ({data.length} 数据点)
      </div>
    </Card>
  );
};

// ============ SeverityTag 组件 ============
export interface SeverityTagProps {
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  size?: 'small' | 'default' | 'large';
}

export const SeverityTag: React.FC<SeverityTagProps> = ({
  severity,
  size = 'default',
}) => {
  const colorInfo = getSeverityColor(severity);
  const severityLabels: Record<string, string> = {
    CRITICAL: '严重',
    HIGH: '高',
    MEDIUM: '中',
    LOW: '低',
    INFO: '信息',
  };

  return (
    <Tag
      color={colorInfo.color}
      style={{
        padding:
          size === 'small'
            ? '2px 8px'
            : size === 'large'
              ? '6px 16px'
              : '4px 12px',
        fontSize: size === 'small' ? 12 : size === 'large' ? 14 : 13,
        fontWeight: 500,
        border: 'none',
      }}
    >
      {severityLabels[severity]}
    </Tag>
  );
};

// ============ StatusBadge 组件 ============
export interface StatusBadgeProps {
  status:
    | 'success'
    | 'error'
    | 'warning'
    | 'processing'
    | 'default';
  label: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
}) => {
  const statusConfig = {
    success: { color: '#12B76A', text: '成功' },
    error: { color: '#F04438', text: '错误' },
    warning: { color: '#F79009', text: '警告' },
    processing: { color: '#2E90FA', text: '处理中' },
    default: { color: '#64748B', text: '默认' },
  };

  const config = statusConfig[status];

  return (
    <Badge
      color={config.color}
      text={label || config.text}
      style={{
        fontSize: fontSize.sm,
        fontWeight: 500,
      }}
    />
  );
};

// ============ EmptyState 组件 ============
export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
}) => {
  return (
    <EmptyStateWrapper>
      {icon && <div className="empty-icon">{icon}</div>}
      <div className="empty-title">{title}</div>
      {description && (
        <div className="empty-description">{description}</div>
      )}
      {action && <div>{action}</div>}
    </EmptyStateWrapper>
  );
};

// ============ LoadingSkeleton 组件 ============
export interface LoadingSkeletonProps {
  count?: number;
  rows?: number;
  type?: 'card' | 'list' | 'table';
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  count = 1,
  rows = 3,
  type = 'card',
}) => {
  const skeletons = Array.from({ length: count }, (_, i) => (
    <div key={i} style={{ marginBottom: spacing.md }}>
      {type === 'card' && (
        <Card>
          <Skeleton active paragraph={{ rows }} />
        </Card>
      )}
      {type === 'list' && <Skeleton active paragraph={{ rows }} />}
      {type === 'table' && (
        <Skeleton active paragraph={{ rows: rows * 2 }} />
      )}
    </div>
  ));

  return <div>{skeletons}</div>;
};

// ============ ErrorBoundary 组件 ============
export interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  { hasError: boolean; error?: Error }
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <Card
            style={{
              borderRadius: borderRadius.sm,
              border: `1px solid ${getSeverityColor('HIGH').color}`,
            }}
          >
            <div style={{ textAlign: 'center', padding: spacing.lg }}>
              <ExclamationCircleOutlined
                style={{
                  fontSize: 48,
                  color: getSeverityColor('HIGH').color,
                  marginBottom: spacing.md,
                }}
              />
              <h3>出错了</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                {this.state.error?.message ||
                  '发生了意外错误，请刷新页面重试'}
              </p>
              <Button onClick={() => window.location.reload()}>
                刷新页面
              </Button>
            </div>
          </Card>
        )
      );
    }

    return this.props.children;
  }
}

// ============ SSEStatusBar 组件 ============
export interface SSEStatusBarProps {
  taskName: string;
  progress: number;
  cacheHitRatio: number;
  eta: number; // 剩余秒数
}

export const SSEStatusBar: React.FC<SSEStatusBarProps> = ({
  taskName,
  progress,
  cacheHitRatio,
  eta,
}) => {
  const formatETA = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m${seconds % 60}s`;
  };

  return (
    <SSEStatusBarWrapper>
      <div className="task-info">
        <Spin size="small" />
        <span>{taskName}</span>
      </div>
      <div className="metrics">
        <div className="metric">
          <div className="label">进度</div>
          <div className="value">{progress}%</div>
        </div>
        <div className="metric">
          <div className="label">缓存命中率</div>
          <div className="value">
            {(cacheHitRatio * 100).toFixed(1)}%
          </div>
        </div>
        <div className="metric">
          <div className="label">预计耗时</div>
          <div className="value">{formatETA(eta)}</div>
        </div>
      </div>
    </SSEStatusBarWrapper>
  );
};
