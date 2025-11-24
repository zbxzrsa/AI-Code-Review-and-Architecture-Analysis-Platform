import React from 'react';
import { Empty, Button } from 'antd';
import styled from 'styled-components';
import { spacing } from '../../styles/tokens';

const Wrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${spacing.xl}px;
`;

interface EmptyStateProps {
  title?: string;
  description?: React.ReactNode;
  actionLabel?: string;
  onAction?: () => void;
}

const EmptyState: React.FC<EmptyStateProps> = ({ title = '暂无内容', description, actionLabel, onAction }) => (
  <Wrapper>
    <Empty description={description || '没有可显示的数据'}>
      {actionLabel && <Button type="primary" onClick={onAction}>{actionLabel}</Button>}
    </Empty>
  </Wrapper>
);

export default EmptyState;
