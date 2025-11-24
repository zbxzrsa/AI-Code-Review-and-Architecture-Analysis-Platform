/**
 * LoadingSkeleton 组件
 * 用于列表、卡片加载时显示占位符，避免布局跳动
 */

import React from 'react';
import { Skeleton, Row, Col } from 'antd';
import styled from 'styled-components';
import { spacing, borderRadius } from '../../styles/tokens';

const SkeletonCard = styled.div`
  padding: ${spacing.lg}px;
  border-radius: ${borderRadius.sm}px;
  background: var(--bg-base);
  border: 1px solid var(--border-default);
  margin-bottom: ${spacing.md}px;
`;

export const CardSkeleton: React.FC<{ count?: number }> = ({ count = 3 }) => {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i}>
          <Skeleton active paragraph={{ rows: 2 }} />
        </SkeletonCard>
      ))}
    </>
  );
};

export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({ rows = 5, columns = 4 }) => {
  return (
    <div>
      {Array.from({ length: rows }).map((_, i) => (
        <Row key={i} gutter={16} style={{ marginBottom: spacing.md }}>
          {Array.from({ length: columns }).map((_, j) => (
            <Col key={j} span={24 / columns}>
              <Skeleton active />
            </Col>
          ))}
        </Row>
      ))}
    </div>
  );
};

export const ListItemSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => {
  return (
    <div style={{ padding: spacing.md }}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{ marginBottom: spacing.md }}>
          <Skeleton avatar active paragraph={{ rows: 1 }} />
        </div>
      ))}
    </div>
  );
};

export default CardSkeleton;
