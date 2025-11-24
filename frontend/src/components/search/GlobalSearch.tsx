import React, { useState, useEffect } from 'react';
import { Input, AutoComplete, Card, List, Typography, Space, Tag, Button } from 'antd';
import { SearchOutlined, FileSearchOutlined, ProjectOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Search } = Input;
const { Text, Title } = Typography;

interface SearchResult {
  id: string;
  type: 'project' | 'session' | 'issue' | 'file';
  title: string;
  description: string;
  path: string;
  timestamp?: string;
  tags?: string[];
}

const GlobalSearch: React.FC = () => {
  const navigate = useNavigate();
  const [searchValue, setSearchValue] = useState('');
  const [options, setOptions] = useState<{ value: string; label: React.ReactNode }[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);

  // Mock data for search
  const mockData: SearchResult[] = [
    {
      id: '1',
      type: 'project',
      title: 'frontend-app',
      description: 'React TypeScript frontend application',
      path: '/projects/1',
      tags: ['React', 'TypeScript', 'Frontend']
    },
    {
      id: '2',
      type: 'project',
      title: 'backend-api',
      description: 'Python FastAPI backend service',
      path: '/projects/2',
      tags: ['Python', 'FastAPI', 'Backend']
    },
    {
      id: '3',
      type: 'session',
      title: 'Security Analysis - frontend-app',
      description: 'Comprehensive security vulnerability assessment',
      path: '/sessions/123',
      timestamp: '2 hours ago',
      tags: ['Security', 'Completed']
    },
    {
      id: '4',
      type: 'issue',
      title: 'SQL Injection vulnerability',
      description: 'High severity SQL injection found in user authentication',
      path: '/analysis/issues/456',
      tags: ['Critical', 'Security']
    },
    {
      id: '5',
      type: 'file',
      title: 'auth.service.ts',
      description: 'Authentication service with JWT implementation',
      path: '/projects/1/files/auth.service.ts',
      tags: ['TypeScript', 'Authentication']
    },
    {
      id: '6',
      type: 'session',
      title: 'Performance Analysis - backend-api',
      description: 'Database query optimization analysis',
      path: '/sessions/124',
      timestamp: '1 day ago',
      tags: ['Performance', 'In Progress']
    }
  ];

  useEffect(() => {
    if (searchValue.length > 0) {
      const filteredOptions = mockData
        .filter(item => 
          item.title.toLowerCase().includes(searchValue.toLowerCase()) ||
          item.description.toLowerCase().includes(searchValue.toLowerCase()) ||
          item.tags?.some(tag => tag.toLowerCase().includes(searchValue.toLowerCase()))
        )
        .map(item => ({
          value: item.title,
          label: (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {getTypeIcon(item.type)}
              <div>
                <div>{item.title}</div>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {item.description}
                </Text>
              </div>
            </div>
          )
        }));
      
      setOptions(filteredOptions);
    } else {
      setOptions([]);
    }
  }, [searchValue]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'project': return <ProjectOutlined />;
      case 'session': return <ClockCircleOutlined />;
      case 'issue': return <FileSearchOutlined />;
      default: return <SearchOutlined />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'project': return 'blue';
      case 'session': return 'green';
      case 'issue': return 'red';
      case 'file': return 'orange';
      default: return 'default';
    }
  };

  const handleSearch = (value: string) => {
    setSearchValue(value);
    
    if (value.length > 0) {
      const results = mockData.filter(item => 
        item.title.toLowerCase().includes(value.toLowerCase()) ||
        item.description.toLowerCase().includes(value.toLowerCase()) ||
        item.tags?.some(tag => tag.toLowerCase().includes(value.toLowerCase()))
      );
      
      setSearchResults(results);
      setShowResults(true);
    } else {
      setSearchResults([]);
      setShowResults(false);
    }
  };

  const handleSelect = (value: string) => {
    const selectedItem = mockData.find(item => item.title === value);
    if (selectedItem) {
      navigate(selectedItem.path);
      setShowResults(false);
      setSearchValue('');
    }
  };

  const handleResultClick = (result: SearchResult) => {
    navigate(result.path);
    setShowResults(false);
    setSearchValue('');
  };

  const handleQuickAction = (type: string) => {
    switch (type) {
      case 'projects':
        navigate('/projects');
        break;
      case 'sessions':
        navigate('/sessions');
        break;
      case 'analysis':
        navigate('/analysis');
        break;
      case 'issues':
        navigate('/analysis?tab=issues');
        break;
    }
    setShowResults(false);
  };

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <AutoComplete
        style={{ width: '100%' }}
        options={options}
        onSelect={handleSelect}
        onSearch={handleSearch}
        value={searchValue}
        open={options.length > 0 && !showResults}
      >
        <Search
          placeholder="Search projects, sessions, issues, files..."
          allowClear
          enterButton={<SearchOutlined />}
          size="middle"
          onSearch={handleSearch}
          onFocus={() => {
            if (searchValue.length > 0) {
              setShowResults(true);
            }
          }}
        />
      </AutoComplete>

      {showResults && searchResults.length > 0 && (
        <Card
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            zIndex: 1000,
            maxHeight: '400px',
            overflowY: 'auto',
            marginTop: '8px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}
          bodyStyle={{ padding: 0 }}
        >
          <div style={{ padding: '12px', borderBottom: '1px solid #f0f0f0' }}>
            <Space wrap>
              <Button size="small" onClick={() => handleQuickAction('projects')}>
                Projects
              </Button>
              <Button size="small" onClick={() => handleQuickAction('sessions')}>
                Sessions
              </Button>
              <Button size="small" onClick={() => handleQuickAction('analysis')}>
                Analysis
              </Button>
              <Button size="small" onClick={() => handleQuickAction('issues')}>
                Issues
              </Button>
            </Space>
          </div>
          
          <List
            dataSource={searchResults}
            renderItem={(result) => (
              <List.Item
                style={{
                  padding: '12px 16px',
                  cursor: 'pointer',
                  borderBottom: '1px solid #f0f0f0',
                  transition: 'background-color 0.2s'
                }}
                onClick={() => handleResultClick(result)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f5f5f5';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <List.Item.Meta
                  avatar={getTypeIcon(result.type)}
                  title={
                    <Space>
                      {result.title}
                      <Tag color={getTypeColor(result.type)}>
                        {result.type}
                      </Tag>
                    </Space>
                  }
                  description={
                    <div>
                      <Text type="secondary">{result.description}</Text>
                      {result.timestamp && (
                        <div>
                          <Text type="secondary" style={{ fontSize: '11px' }}>
                            {result.timestamp}
                          </Text>
                        </div>
                      )}
                      {result.tags && (
                        <div style={{ marginTop: '4px' }}>
                          <Space wrap size={4}>
                            {result.tags.map(tag => (
                              <Tag key={tag} color="default">
                                {tag}
                              </Tag>
                            ))}
                          </Space>
                        </div>
                      )}
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* Click outside to close results */}
      {showResults && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 999
          }}
          onClick={() => setShowResults(false)}
        />
      )}
    </div>
  );
};

export default GlobalSearch;