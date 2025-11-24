import React from 'react';
import styled from 'styled-components';
import { Button, Space, Tag } from 'antd';
import { fontSize, spacing, borderRadius } from '../../styles/tokens';

const HeaderWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${spacing.lg}px;
  margin-bottom: ${spacing.lg}px;
`;

const TitleGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${spacing.xs}px;
`;

const Title = styled.h1`
  margin: 0;
  font-size: ${fontSize.xxl}px;
  line-height: 1;
`;

const Subtitle = styled.div`
  color: var(--text-secondary);
  font-size: ${fontSize.base}px;
`;

export interface PageHeaderProps {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  tags?: Array<{ key: string; label: string }>;
  primaryAction?: React.ReactNode;
  extra?: React.ReactNode;
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, subtitle, tags = [], primaryAction, extra }) => {
  return (
    <HeaderWrapper>
      <TitleGroup>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <Title>{title}</Title>
          <Space>
            {tags.map((t) => (
              <Tag key={t.key}>{t.label}</Tag>
            ))}
          </Space>
        </div>
        {subtitle && <Subtitle>{subtitle}</Subtitle>}
      </TitleGroup>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {extra}
        {primaryAction}
      </div>
    </HeaderWrapper>
  );
};

export default PageHeader;
