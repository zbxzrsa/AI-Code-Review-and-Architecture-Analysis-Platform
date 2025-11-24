import React, { useEffect, useState } from 'react';
import { Card, Form, Input, Button, message, Spin, List, Avatar, Typography, Tag, Space } from 'antd';
import { GithubOutlined, StarOutlined, ForkOutlined, BranchesOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
 

const { Title, Text } = Typography;

interface Repository {
  id: number;
  name: string;
  full_name: string;
  html_url: string;
  description: string;
  stargazers_count: number;
  forks_count: number;
  language: string;
}

const GitHubConnect: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [token, setToken] = useState('');
  const [username, setUsername] = useState('');
  const [connected, setConnected] = useState(false);
  const [oauthCode, setOauthCode] = useState<string | null>(null);

  const clientId = process.env.REACT_APP_GITHUB_CLIENT_ID || 'demo-client-id';
  const redirectUri = `${window.location.origin}/github-connect`;

  const handleConnect = () => {
    const targetUsername = username || 'demo-user';
    if (!username) {
      setUsername(targetUsername);
    }

    setLoading(true);

    // Simulated API call
    setTimeout(() => {
      // Simulated repositories
      const mockRepositories: Repository[] = [
        {
          id: 1,
          name: 'code-review-ai',
          full_name: `${targetUsername}/code-review-ai`,
          html_url: `https://github.com/${targetUsername}/code-review-ai`,
          description: 'Intelligent code review tool using AI to detect issues',
          stargazers_count: 128,
          forks_count: 32,
          language: 'TypeScript'
        },
        {
          id: 2,
          name: 'architecture-analyzer',
          full_name: `${targetUsername}/architecture-analyzer`,
          html_url: `https://github.com/${targetUsername}/architecture-analyzer`,
          description: 'Software architecture analyzer to help understand complex systems',
          stargazers_count: 85,
          forks_count: 17,
          language: 'Python'
        },
        {
          id: 3,
          name: 'security-scanner',
          full_name: `${targetUsername}/security-scanner`,
          html_url: `https://github.com/${targetUsername}/security-scanner`,
          description: 'Code security scanner detecting potential vulnerabilities',
          stargazers_count: 56,
          forks_count: 12,
          language: 'JavaScript'
        }
      ];

      setRepositories(mockRepositories);
      setConnected(true);
      setLoading(false);
      message.success(`Connected to GitHub user ${targetUsername}`);
    }, 1500);
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const codeParam = params.get('code');
    if (codeParam && !connected) {
      setOauthCode(codeParam);
      message.success('OAuth code captured. Exchanging token...');
      setLoading(true);
      setTimeout(() => {
        handleConnect();
        setLoading(false);
      }, 1000);
    }
  }, [connected]);

  const handleAnalyzeRepository = (repo: Repository) => {
    navigate(`/analysis?repo=${encodeURIComponent(repo.full_name)}`);
  };

  const getLanguageColor = (language: string) => {
    const colors: Record<string, string> = {
      'TypeScript': '#3178c6',
      'JavaScript': '#f1e05a',
      'Python': '#3572A5',
      'Java': '#b07219',
      'Go': '#00ADD8',
      'C#': '#178600',
      'PHP': '#4F5D95',
      'Ruby': '#701516',
      'Swift': '#F05138',
      'Kotlin': '#A97BFF'
    };
    
    return colors[language] || '#8e8e8e';
  };

  return (
    <div>
      <Title level={3}>
        <GithubOutlined /> GitHub Connect
      </Title>
      <Typography.Paragraph type="secondary">
        Authenticate with OAuth, sync repositories, and trigger automated analysis without
        leaving the platform.
      </Typography.Paragraph>

      {!connected ? (
        <Card>
          <Form layout="vertical">
            <Form.Item 
              label="GitHub Username"
              required
            >
              <Input 
                placeholder="Enter GitHub username"
                value={username}
                onChange={e => setUsername(e.target.value)}
                prefix={<GithubOutlined />}
              />
            </Form.Item>
            
            <Form.Item label="GitHub Access Token (optional)">
              <Input.Password 
                placeholder="Used to access private repositories"
                value={token}
                onChange={e => setToken(e.target.value)}
              />
            </Form.Item>

            <Form.Item label="OAuth flow">
              <Button
                icon={<GithubOutlined />}
                href={`https://github.com/login/oauth/authorize?client_id=${clientId}&redirect_uri=${encodeURIComponent(redirectUri)}&scope=repo`}
              >
                Authenticate with GitHub
              </Button>
              {oauthCode && (
                <Typography.Paragraph type="secondary" style={{ marginTop: 8 }}>
                  OAuth code: {oauthCode}
                </Typography.Paragraph>
              )}
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                onClick={handleConnect}
                loading={loading}
                icon={<GithubOutlined />}
              >
                Connect to GitHub
              </Button>
            </Form.Item>
          </Form>
        </Card>
      ) : (
        <>
          <Card style={{ marginBottom: 16 }}>
            <Space>
              <Avatar size={64} icon={<GithubOutlined />} />
              <div>
                <Title level={4}>{username}</Title>
                <Button 
                  type="link" 
                  onClick={() => setConnected(false)}
                >
                  Disconnect
                </Button>
              </div>
            </Space>
          </Card>
          
          <Title level={4}>Repository List</Title>
          
          {loading ? (
            <div style={{ textAlign: 'center', padding: '30px 0' }}>
              <Spin size="large" />
            </div>
          ) : (
            <List
              itemLayout="vertical"
              dataSource={repositories}
              renderItem={repo => (
                <List.Item
                  actions={[
                    <Space key="stars">
                      <StarOutlined /> {repo.stargazers_count}
                    </Space>,
                    <Space key="forks">
                      <ForkOutlined /> {repo.forks_count}
                    </Space>,
                    <Button 
                      key="analyze" 
                      type="primary"
                      onClick={() => handleAnalyzeRepository(repo)}
                      icon={<BranchesOutlined />}
                    >
                      {'Analyze Repository'}
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <a href={repo.html_url} target="_blank" rel="noopener noreferrer">
                        {repo.full_name}
                      </a>
                    }
                    description={
                      <Space direction="vertical">
                        <Text>{repo.description}</Text>
                        <Tag color={getLanguageColor(repo.language)}>
                          {repo.language}
                        </Tag>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          )}
        </>
      )}

      <Card style={{ marginTop: 16 }} title="Example demonstrations">
        <List
          dataSource={[
            'Use OAuth to sync repositories and trigger automated baselines.',
            'Switch between providers using the AI chat dock after connecting GitHub.',
            'Export repository metadata to Projects to create guardrails automatically.',
          ]}
          renderItem={(item) => (
            <List.Item>
              <Text>{item}</Text>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default GitHubConnect;