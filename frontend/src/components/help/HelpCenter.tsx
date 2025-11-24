import React, { useState } from 'react';
import { Card, Input, Button, Typography, Collapse, List, Tag, Space, Divider, Row, Col } from 'antd';
import { SearchOutlined, QuestionCircleOutlined, BookOutlined, VideoCameraOutlined, MessageOutlined, RocketOutlined } from '@ant-design/icons';
import MicroInteractions from '../ui/MicroInteractions';

const { Title, Paragraph, Text, Link } = Typography;
const { Panel } = Collapse;
const { FadeIn } = MicroInteractions;

// Frequently Asked Questions data
const faqs = [
  {
    question: 'How do I start my first code analysis?',
    answer: 'Click "New Analysis", choose your repository or upload files, select analysis type and scope, then click "Start".',
    tags: ['Getting Started', 'Code Analysis']
  },
  {
    question: 'How do I understand the code quality score?',
    answer: 'The score aggregates complexity, duplication, testability, and potential issues from 0–100. Higher is better.',
    tags: ['Quality', 'Metrics']
  },
  {
    question: 'How do I export the analysis report?',
    answer: 'On the results page, click "Export", choose PDF/HTML/Excel, and confirm to download.',
    tags: ['Report', 'Export']
  },
  {
    question: 'How do I share results with my team?',
    answer: 'On the results page, click "Share", enter teammates’ emails or usernames, set permissions, and send invites.',
    tags: ['Collaboration', 'Share']
  },
  {
    question: 'How do I customize analysis rules?',
    answer: 'Open "Settings" → "Analysis Rules" to enable/disable rules, adjust severity, or create custom ones.',
    tags: ['Customization', 'Rules']
  }
];

// Video tutorials data
const videoTutorials = [
  {
    title: 'Quick Platform Overview',
    description: 'Understand core features and basic operations in 5 minutes',
    duration: '5:20',
    url: '#/tutorials/getting-started'
  },
  {
    title: 'Code Quality Analysis Explained',
    description: 'Deep dive into metrics and optimization suggestions',
    duration: '8:45',
    url: '#/tutorials/code-quality'
  },
  {
    title: 'Architecture Visualization Tutorial',
    description: 'Learn how to analyze code structure with visualization',
    duration: '7:15',
    url: '#/tutorials/architecture-visualization'
  },
  {
    title: 'Team Collaboration Best Practices',
    description: 'Collaborate efficiently on code reviews',
    duration: '6:30',
    url: '#/tutorials/team-collaboration'
  }
];

// Documentation resources data
const documentationResources = [
  {
    title: 'User Manual',
    description: 'Complete platform features and operation guide',
    url: '#/docs/user-manual'
  },
  {
    title: 'API Documentation',
    description: 'API references and examples',
    url: '#/docs/api'
  },
  {
    title: 'Best Practices Guide',
    description: 'Best practices for code review and architecture analysis',
    url: '#/docs/best-practices'
  },
  {
    title: 'FAQ',
    description: 'Detailed frequently asked questions',
    url: '#/docs/faq'
  }
];

interface HelpCenterProps {
  onContactSupport?: () => void;
}

const HelpCenter: React.FC<HelpCenterProps> = ({ onContactSupport }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredFaqs, setFilteredFaqs] = useState(faqs);

  // Search handler
  const handleSearch = (value: string) => {
    setSearchQuery(value);
    if (!value) {
      setFilteredFaqs(faqs);
      return;
    }
    
    const lowerCaseQuery = value.toLowerCase();
    const filtered = faqs.filter(
      faq => 
        faq.question.toLowerCase().includes(lowerCaseQuery) || 
        faq.answer.toLowerCase().includes(lowerCaseQuery) ||
        faq.tags.some(tag => tag.toLowerCase().includes(lowerCaseQuery))
    );
    
    setFilteredFaqs(filtered);
  };

  return (
    <div className="help-center">
      <FadeIn>
        <Card className="help-search-card" style={{ marginBottom: 24 }}>
          <Title level={4}>
            <QuestionCircleOutlined /> Help Center
          </Title>
          <Paragraph>
            Find the help and support resources you need and resolve issues quickly.
          </Paragraph>
          
          <Input.Search
            placeholder="Search questions, tutorials, or docs..."
            allowClear
            enterButton={<SearchOutlined />}
            size="large"
            value={searchQuery}
            onChange={e => handleSearch(e.target.value)}
            onSearch={handleSearch}
            style={{ marginBottom: 16 }}
          />
          
          <div className="quick-help-buttons" style={{ display: 'flex', gap: 8 }}>
            <Button icon={<BookOutlined />}>User Manual</Button>
            <Button icon={<VideoCameraOutlined />}>Video Tutorials</Button>
            <Button 
              type="primary" 
              icon={<MessageOutlined />}
              onClick={onContactSupport}
            >
              Contact Support
            </Button>
          </div>
        </Card>
        
        <Row gutter={[24, 24]}>
          <Col xs={24} lg={16}>
            {/* FAQs */}
            <Card title={<><QuestionCircleOutlined /> Frequently Asked Questions</>} style={{ marginBottom: 24 }}>
              {filteredFaqs.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '20px 0' }}>
                  <Text type="secondary">No matching questions. Try other keywords or contact support.</Text>
                </div>
              ) : (
                <Collapse accordion>
                  {filteredFaqs.map((faq, index) => (
                    <Panel 
                      header={faq.question} 
                      key={index}
                      extra={
                        <Space>
                          {faq.tags.map(tag => (
                            <Tag key={tag} color="blue">{tag}</Tag>
                          ))}
                        </Space>
                      }
                    >
                      <Paragraph>{faq.answer}</Paragraph>
                      <div style={{ textAlign: 'right' }}>
                        <Text type="secondary">Did this solve your problem?</Text>
                        <Space style={{ marginLeft: 8 }}>
                          <Button size="small" type="text">Yes</Button>
                          <Button size="small" type="text">No</Button>
                        </Space>
                      </div>
                    </Panel>
                  ))}
                </Collapse>
              )}
              
              <div style={{ textAlign: 'center', marginTop: 16 }}>
                <Button type="link">View more FAQs</Button>
              </div>
            </Card>
            
            {/* Video Tutorials */}
            <Card title={<><VideoCameraOutlined /> Video Tutorials</>} style={{ marginBottom: 24 }}>
              <List
                itemLayout="horizontal"
                dataSource={videoTutorials}
                renderItem={item => (
                  <List.Item
                    actions={[
                      <Button type="link" key="watch">Watch</Button>,
                      <Text type="secondary" key="duration">{item.duration}</Text>
                    ]}
                  >
                    <List.Item.Meta
                      avatar={<VideoCameraOutlined style={{ fontSize: 24 }} />}
                      title={<Link href={item.url}>{item.title}</Link>}
                      description={item.description}
                    />
                  </List.Item>
                )}
              />
              
              <div style={{ textAlign: 'center', marginTop: 16 }}>
                <Button type="link">View all tutorials</Button>
              </div>
            </Card>
          </Col>
          
          <Col xs={24} lg={8}>
            {/* Quick Start */}
            <Card title="Quick Start" style={{ marginBottom: 24 }}>
              <div style={{ padding: '8px 0' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button type="link" icon={<RocketOutlined />} block style={{ textAlign: 'left' }}>
                    Create your first project
                  </Button>
                  <Divider style={{ margin: '8px 0' }} />
                  <Button type="link" icon={<SearchOutlined />} block style={{ textAlign: 'left' }}>
                    Run code analysis
                  </Button>
                  <Divider style={{ margin: '8px 0' }} />
                  <Button type="link" icon={<BookOutlined />} block style={{ textAlign: 'left' }}>
                    Understand analysis results
                  </Button>
                  <Divider style={{ margin: '8px 0' }} />
                  <Button type="link" icon={<MessageOutlined />} block style={{ textAlign: 'left' }}>
                    Share and collaborate
                  </Button>
                </Space>
              </div>
            </Card>
            
            {/* Documentation */}
            <Card title={<><BookOutlined /> Documentation</>}>
              <List
                itemLayout="horizontal"
                dataSource={documentationResources}
                renderItem={item => (
                  <List.Item
                    actions={[
                      <Button type="link" key="view">View</Button>
                    ]}
                  >
                    <List.Item.Meta
                      title={<Link href={item.url}>{item.title}</Link>}
                      description={item.description}
                    />
                  </List.Item>
                )}
              />
            </Card>
            
            {/* Contact Support */}
            <Card 
              title="Need more help?" 
              style={{ marginTop: 24, textAlign: 'center' }}
              actions={[
                <Button type="primary" key="contact" onClick={onContactSupport}>
                  Contact Support
                </Button>
              ]}
            >
              <Paragraph>
                Our support team is ready to help. Average response time is under 2 hours.
              </Paragraph>
              <div style={{ margin: '16px 0' }}>
                <Text strong>Working Hours:</Text> Mon–Fri 9:00–18:00
              </div>
            </Card>
          </Col>
        </Row>
      </FadeIn>
    </div>
  );
};

export default HelpCenter;