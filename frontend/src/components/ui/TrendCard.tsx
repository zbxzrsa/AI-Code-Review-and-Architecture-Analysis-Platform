import React from 'react';
import { Card } from 'antd';
import styled from 'styled-components';
import { componentStyles, fontSize } from '../../styles/tokens';

const StyledCard = styled(Card)`
  border-radius: ${componentStyles.card.borderRadius}px;
  box-shadow: ${componentStyles.card.boxShadow};
  .ant-card-body { padding: ${componentStyles.card.padding}px; }
`;

const Sparkline = styled.div`
  height: 40px;
  width: 100%;
  background: linear-gradient(90deg, rgba(46,144,250,0.08), transparent);
  border-radius: 4px;
`;

interface TrendCardProps {
  title: React.ReactNode;
  value: React.ReactNode;
  sparkData?: number[];
}

const TrendCard: React.FC<TrendCardProps> = ({ title, value, sparkData }) => {
  return (
    <StyledCard>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{title}</div>
          <div style={{ fontSize: `${fontSize.xxl}px`, fontWeight: 600 }}>{value}</div>
        </div>
        <div style={{ width: 120 }}>
          <Sparkline aria-hidden />
        </div>
      </div>
    </StyledCard>
  );
};

export default TrendCard;
