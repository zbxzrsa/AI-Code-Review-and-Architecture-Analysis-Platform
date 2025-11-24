import React from 'react';
import { Layout, Typography, Breadcrumb } from 'antd';
import { HomeOutlined } from '@ant-design/icons';
import { Link, useNavigate } from 'react-router-dom';
import QuickStartTemplates from '../components/onboarding/QuickStartTemplates';
 

const { Content } = Layout;
const { Title } = Typography;

const QuickStartPage: React.FC = () => {
  const t = (k: string, fb?: string) => fb || k;
  const navigate = useNavigate();

  return (
    <Content style={{ padding: '0 24px', minHeight: 280 }}>
      <Breadcrumb style={{ margin: '16px 0' }}>
        <Breadcrumb.Item>
          <Link to="/">
            <HomeOutlined /> {t('breadcrumb.home', 'Home')}
          </Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>{t('breadcrumb.quick_start', 'Quick Start')}</Breadcrumb.Item>
      </Breadcrumb>
      
      <div className="site-layout-content" style={{ padding: 24, background: '#fff', borderRadius: 8 }}>
        <Title level={2}>{t('quick_start.title', 'Quick Start')}</Title>
        <Typography.Paragraph>
          {t('quick_start.description', 'Select a preconfigured template to quickly start your code analysis project. These templates are optimized for different scenarios to help you get insights faster.')}
        </Typography.Paragraph>
        
        <QuickStartTemplates onSelectTemplate={(id) => navigate('/analysis')} />
      </div>
    </Content>
  );
};

export default QuickStartPage;
