import React from 'react';
import { Card, Row, Col, Button, Typography, Tag, Space, Divider } from 'antd';
import { RocketOutlined, CodeOutlined, BranchesOutlined, BugOutlined, StarOutlined } from '@ant-design/icons';
import MicroInteractions from '../ui/MicroInteractions';

const { Title, Paragraph, Text } = Typography;
const { FadeIn } = MicroInteractions;

// 模板类型定义
export interface QuickStartTemplate {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  popularity: number;
}

  // Example template data
const TEMPLATES: QuickStartTemplate[] = [
  {
    id: 'basic-code-quality',
    title: 'Basic Code Quality Check',
    description: 'Quickly check common issues and quality metrics; great for beginners.',
    icon: <CodeOutlined />,
    tags: ['代码质量', '新手友好', '快速分析'],
    difficulty: 'beginner',
    estimatedTime: '5 min',
    popularity: 95
  },
  {
    id: 'architecture-overview',
    title: 'Project Architecture Overview',
    description: 'Generate architecture diagrams and dependency analysis to understand structure.',
    icon: <BranchesOutlined />,
    tags: ['架构分析', '依赖关系', '可视化'],
    difficulty: 'intermediate',
    estimatedTime: '10 min',
    popularity: 85
  },
  {
    id: 'security-scan',
    title: 'Security Vulnerability Scan',
    description: 'Detect potential security vulnerabilities and risks, with remediation suggestions.',
    icon: <BugOutlined />,
    tags: ['安全', '漏洞检测', '风险评估'],
    difficulty: 'intermediate',
    estimatedTime: '15 min',
    popularity: 90
  },
  {
    id: 'performance-optimization',
    title: 'Performance Optimization Analysis',
    description: 'Identify performance bottlenecks and optimization opportunities to improve apps.',
    icon: <RocketOutlined />,
    tags: ['性能优化', '瓶颈分析', '高级'],
    difficulty: 'advanced',
    estimatedTime: '20 min',
    popularity: 80
  }
];

interface QuickStartTemplatesProps {
  onSelectTemplate: (templateId: string) => void;
}

const QuickStartTemplates: React.FC<QuickStartTemplatesProps> = ({ onSelectTemplate }) => {
  // 获取难度标签颜色
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner':
        return 'success';
      case 'intermediate':
        return 'processing';
      case 'advanced':
        return 'warning';
      default:
        return 'default';
    }
  };
  
  // Difficulty labels
  const getDifficultyLabel = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner':
        return 'Beginner';
      case 'intermediate':
        return 'Intermediate';
      case 'advanced':
        return 'Advanced';
      default:
        return 'Unknown';
    }
  };
  
  // 渲染人气指标
  const renderPopularity = (popularity: number) => {
    const stars = [];
    const fullStars = Math.floor(popularity / 20);
    
    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(<StarOutlined key={i} style={{ color: '#fadb14' }} />);
      } else {
        stars.push(<StarOutlined key={i} style={{ color: '#d9d9d9' }} />);
      }
    }
    
    return (
      <Space>
        {stars}
        <Text type="secondary">{popularity}% users choose</Text>
      </Space>
    );
  };

  return (
    <div className="quick-start-templates">
      <FadeIn>
        <Title level={4}>
          <RocketOutlined /> Quick Start Templates
        </Title>
        <Paragraph>
          Choose a preconfigured template to quickly start code analysis with best defaults.
        </Paragraph>
        
        <Row gutter={[16, 16]}>
          {TEMPLATES.map(template => (
            <Col xs={24} sm={12} md={8} lg={6} key={template.id}>
              <Card 
                hoverable 
                className="template-card"
                actions={[
                  <Button 
                    type="primary" 
                    key="use" 
                    onClick={() => onSelectTemplate(template.id)}
                  >
                    Use this template
                  </Button>
                ]}
              >
                <div style={{ fontSize: 32, color: '#1677ff', marginBottom: 16, textAlign: 'center' }}>
                  {template.icon}
                </div>
                
                <Title level={5}>{template.title}</Title>
                <Paragraph ellipsis={{ rows: 2 }}>{template.description}</Paragraph>
                
                <Space wrap style={{ marginBottom: 12 }}>
                  <Tag color={getDifficultyColor(template.difficulty)}>
                    {getDifficultyLabel(template.difficulty)}
                  </Tag>
                  <Tag color="blue">{template.estimatedTime}</Tag>
                </Space>
                
                <Divider style={{ margin: '12px 0' }} />
                
                <div style={{ marginBottom: 8 }}>
                  {template.tags.map(tag => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </div>
                
                <div>{renderPopularity(template.popularity)}</div>
              </Card>
            </Col>
          ))}
        </Row>
        
        <div style={{ textAlign: 'center', marginTop: 24 }}>
          <Button type="link">View more templates</Button>
        </div>
      </FadeIn>
    </div>
  );
};

export default QuickStartTemplates;