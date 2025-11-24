import React from 'react';
import { Card } from 'antd';
import styled from 'styled-components';
import { componentStyles, fontSize, spacing } from '../../styles/tokens';

const StyledCard = styled(Card)`
  border-radius: ${componentStyles.card.borderRadius}px;
  box-shadow: ${componentStyles.card.boxShadow};
  .ant-card-body {
    padding: ${componentStyles.card.padding}px;
  }
`;

interface StatCardProps {
  title: React.ReactNode;
  value: React.ReactNode;
  subtitle?: React.ReactNode;
}

const Value = styled.div`
  font-size: ${fontSize.xxl}px;
  font-weight: 600;
`;

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle }) => (
  <StyledCard>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div>
        <div style={{ color: 'var(--text-secondary)', fontSize: `${fontSize.sm}px` }}>{title}</div>
        <Value>{value}</Value>
        {subtitle && <div style={{ color: 'var(--text-secondary)', marginTop: `${spacing.xs}px` }}>{subtitle}</div>}
      </div>
    </div>
  </StyledCard>
);

export default StatCard;
