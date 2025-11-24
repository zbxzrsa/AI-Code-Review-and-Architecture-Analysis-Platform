import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Form,
  Input,
  Button,
  Tabs,
  Select,
  Typography,
  Space,
  Card,
  Tag,
  List,
  Alert,
  Progress,
  Divider,
  Badge,
} from 'antd';
import {
  FolderOpenOutlined,
  ThunderboltOutlined,
  GithubOutlined,
  CodeOutlined,
  ConsoleSqlOutlined,
  BranchesOutlined,
  BugOutlined,
  SafetyOutlined,
  BarChartOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons';
import AIChatPanel from '../components/ai/AIChatPanel';
import CodeReviewWorkbench from '../components/codeReview/CodeReviewWorkbench';

const { TextArea } = Input;
const { Title, Paragraph, Text } = Typography;

interface AnalysisResult {
  quality?: {
    score: number;
    issues: Array<{
      type: string;
      severity: string;
      message: string;
      line?: number;
    }>;
    metrics: {
      complexity: number;
      maintainability: number;
      testability: number;
    };
  };
  security?: {
    vulnerabilities: Array<{
      type: string;
      severity: string;
      description: string;
      location: string;
      recommendation: string;
    }>;
  };
  architecture?: {
    patterns: Array<{
      pattern_name: string;
      confidence: number;
      components: string[];
      description: string;
    }>;
    smells: Array<{
      smell_type: string;
      severity: string;
      affected_components: string[];
      description: string;
      recommendation: string;
    }>;
  };
}

type ResizePanel = 'explorer' | 'copilot' | 'bottom';

const explorerSections = [
  {
    title: 'Projects',
    items: ['core-service', 'rules-engine', 'ai-pipeline'],
  },
  {
    title: 'Sessions',
    items: ['session-0421', 'session-0420', 'session-0418'],
  },
  {
    title: 'Artifacts',
    items: ['analysis-report.md', 'trace-log.json', 'baseline.diff'],
  },
];

const terminalLines = [
  'yarn lint --fix',
  '✔ structure graph generated in 2.3s',
  '✔ static analysis completed for 42 files',
  '⚠ 1 security warning surfaced (HIGH)',
  'ℹ use `ai summarize` to send the current run to Copilot',
];

const CodeAnalysis: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('javascript');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [gitUrl, setGitUrl] = useState('');
  const [gitToken, setGitToken] = useState('');
  const [explorerWidth, setExplorerWidth] = useState(260);
  const [copilotWidth, setCopilotWidth] = useState(400);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(220);

  const resizeRef = useRef<{
    panel: ResizePanel | null;
    startX: number;
    startY: number;
    explorer: number;
    copilot: number;
    bottom: number;
  }>({
    panel: null,
    startX: 0,
    startY: 0,
    explorer: 260,
    copilot: 400,
    bottom: 220,
  });

  const handleMouseMove = useCallback((event: MouseEvent) => {
    const meta = resizeRef.current;
    if (!meta.panel) return;

    if (meta.panel === 'explorer') {
      const delta = event.clientX - meta.startX;
      setExplorerWidth(Math.min(420, Math.max(180, meta.explorer + delta)));
    } else if (meta.panel === 'copilot') {
      const delta = meta.startX - event.clientX;
      setCopilotWidth(Math.min(640, Math.max(350, meta.copilot + delta)));
    } else if (meta.panel === 'bottom') {
      const delta = event.clientY - meta.startY;
      setBottomPanelHeight(Math.min(420, Math.max(160, meta.bottom + delta)));
    }
  }, []);

  const stopResizing = useCallback(() => {
    if (!resizeRef.current.panel) return;
    resizeRef.current.panel = null;
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', stopResizing);
  }, [handleMouseMove]);

  const startResizing = useCallback(
    (panel: ResizePanel) => (event: React.MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      resizeRef.current = {
        panel,
        startX: event.clientX,
        startY: event.clientY,
        explorer: explorerWidth,
        copilot: copilotWidth,
        bottom: bottomPanelHeight,
      };
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', stopResizing);
    },
    [bottomPanelHeight, copilotWidth, explorerWidth, handleMouseMove, stopResizing]
  );

  useEffect(
    () => () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', stopResizing);
    },
    [handleMouseMove, stopResizing]
  );

  const runMockAnalysis = () => {
      const mockResult: AnalysisResult = {
        quality: {
        score: 82,
          issues: [
          {
            type: 'Complexity',
            severity: 'medium',
            message: 'Function `normalizeRules` exceeds the recommended cyclomatic threshold.',
            line: 48,
          },
          {
            type: 'Duplication',
            severity: 'high',
            message: 'Duplicate guard clauses detected in `pipeline.ts`.',
            line: 97,
          },
          {
            type: 'Naming',
            severity: 'low',
            message: 'Consider renaming `tmpRes` to a more descriptive identifier.',
            line: 131,
          },
          ],
          metrics: {
          complexity: 68,
          maintainability: 77,
          testability: 83,
        },
        },
        security: {
          vulnerabilities: [
            {
            type: 'Injection',
              severity: 'high',
            description: 'Detected unsanitized template literals inside SQL builder.',
            location: 'src/data/queryBuilder.ts:27-39',
            recommendation: 'Apply parameterized statements via the shared DB client.',
          },
          {
            type: 'Secrets',
              severity: 'medium',
            description: 'A potential AWS access key pattern was found in env.sample.',
            location: 'config/env.sample',
            recommendation: 'Rotate the key and reference it via vaulted secrets.',
          },
        ],
        },
        architecture: {
          patterns: [
            {
            pattern_name: 'Event Sourcing',
            confidence: 0.81,
            components: ['events/', 'listeners/', 'snapshots/'],
            description: 'Commands emit domain events captured in listeners with replay support.',
            },
            {
              pattern_name: 'Repository',
            confidence: 0.74,
            components: ['repositories/', 'entities/'],
            description: 'Domain entities are persisted behind repository boundaries.',
          },
          ],
          smells: [
            {
            smell_type: 'Cyclic Dependency',
              severity: 'high',
            affected_components: ['SessionService', 'BaselineService'],
            description: 'Services import each other, causing an inversion violation.',
            recommendation: 'Extract shared utilities into `analysis-kernel` and inject as needed.',
          },
        ],
      },
    };
      setAnalysisResult(mockResult);
      setLoading(false);
  };

  const onFinish = () => {
    setLoading(true);
    setTimeout(runMockAnalysis, 1500);
  };

  const handleGitAnalysis = () => {
    if (!gitUrl.trim()) return;
    setLoading(true);
    setTimeout(runMockAnalysis, 2000);
  };

  const renderQualityResults = () => {
    if (!analysisResult?.quality) return null;
    const { score, issues, metrics } = analysisResult.quality;
    return (
      <div>
        <Title level={4}>Quality score: {score}/100</Title>
        <Progress percent={score} status={score < 60 ? 'exception' : 'normal'} strokeColor={score > 80 ? '#52c41a' : score > 60 ? '#faad14' : '#f5222d'} />
        <Divider />
        <Title level={5}>Key metrics</Title>
        <div style={{ display: 'flex', gap: 16, marginBottom: 20 }}>
          <Card title="Complexity" style={{ flex: 1 }}>
            <Progress type="dashboard" percent={metrics.complexity} />
          </Card>
          <Card title="Maintainability" style={{ flex: 1 }}>
            <Progress type="dashboard" percent={metrics.maintainability} />
          </Card>
          <Card title="Testability" style={{ flex: 1 }}>
            <Progress type="dashboard" percent={metrics.testability} />
          </Card>
        </div>
        <Divider />
        <Title level={5}>Issues</Title>
        <List
          dataSource={issues}
          renderItem={(item) => (
            <List.Item>
              <List.Item.Meta
                title={
                  <Space>
                    <Tag color={item.severity === 'high' ? 'red' : item.severity === 'medium' ? 'orange' : 'blue'}>
                      {item.severity.toUpperCase()}
                    </Tag>
                    <Text strong>{item.type}</Text>
                    {item.line && <Text type="secondary">Line {item.line}</Text>}
                  </Space>
                }
                description={item.message}
              />
            </List.Item>
          )}
        />
      </div>
    );
  };

  const renderSecurityResults = () => {
    if (!analysisResult?.security) return null;
    return (
      <div>
        <Title level={4}>Security vulnerabilities</Title>
        <List
          itemLayout="vertical"
          dataSource={analysisResult.security.vulnerabilities}
          renderItem={(item) => (
            <List.Item>
              <Card>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Space>
                    <Tag color={item.severity === 'high' ? 'red' : item.severity === 'medium' ? 'orange' : 'blue'}>
                      {item.severity.toUpperCase()}
                    </Tag>
                    <Text strong>{item.type}</Text>
                  </Space>
                  <Paragraph>{item.description}</Paragraph>
                  <Text type="secondary">Location: {item.location}</Text>
                  <Alert message="Recommendation" description={item.recommendation} type="info" />
                </Space>
              </Card>
            </List.Item>
          )}
        />
      </div>
    );
  };

  const renderArchitectureResults = () => {
    if (!analysisResult?.architecture) return null;
    return (
      <div>
        <Title level={4}>Architecture patterns</Title>
        <List
          grid={{ gutter: 16, column: 2 }}
          dataSource={analysisResult.architecture.patterns}
          renderItem={(item) => (
            <List.Item>
              <Card title={item.pattern_name}>
                <Paragraph>{item.description}</Paragraph>
                <Text type="secondary">Confidence: {(item.confidence * 100).toFixed(0)}%</Text>
                <div style={{ marginTop: 10 }}>
                  {item.components.map((component) => (
                    <Tag key={component}>{component}</Tag>
                  ))}
                </div>
              </Card>
            </List.Item>
          )}
        />
        <Divider />
        <Title level={4}>Architecture smells</Title>
        <List
          dataSource={analysisResult.architecture.smells}
          renderItem={(item) => (
            <List.Item>
              <Card>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Space>
                    <Tag color={item.severity === 'high' ? 'red' : item.severity === 'medium' ? 'orange' : 'blue'}>
                      {item.severity.toUpperCase()}
                    </Tag>
                    <Text strong>{item.smell_type}</Text>
                  </Space>
                  <Paragraph>{item.description}</Paragraph>
                  <div>
                    Affected components:
                    {item.affected_components.map((component) => (
                      <Tag key={component} style={{ marginLeft: 4 }}>
                        {component}
                      </Tag>
                    ))}
                  </div>
                  <Alert message="Recommendation" description={item.recommendation} type="info" />
                </Space>
              </Card>
            </List.Item>
          )}
        />
      </div>
    );
  };

  const analysisTabs = [
    {
      key: 'code',
      label: (
              <span>
          <CodeOutlined /> Code input
              </span>
      ),
      children: (
        <Form layout="vertical" onFinish={onFinish}>
          <Form.Item label="Programming language" name="language" initialValue={language}>
            <Select onChange={(value) => setLanguage(value)}>
              <Select.Option value="javascript">JavaScript</Select.Option>
              <Select.Option value="typescript">TypeScript</Select.Option>
              <Select.Option value="python">Python</Select.Option>
              <Select.Option value="java">Java</Select.Option>
              <Select.Option value="go">Go</Select.Option>
              <Select.Option value="csharp">C#</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item
            label="Code content"
                name="code"
            rules={[{ required: true, message: 'Please paste source code to analyze.' }]}
              >
            <TextArea rows={8} placeholder="Paste your code here..." />
              </Form.Item>
                <Button type="primary" htmlType="submit" loading={loading}>
            Analyze code
                </Button>
            </Form>
      ),
    },
    {
      key: 'github',
      label: (
              <span>
          <GithubOutlined /> GitHub repository
              </span>
      ),
      children: (
            <Form layout="vertical">
          <Form.Item label="Repository URL" required>
            <Input placeholder="https://github.com/owner/repo" value={gitUrl} onChange={(event) => setGitUrl(event.target.value)} />
          </Form.Item>
          <Form.Item label="Access token (optional)">
            <Input.Password placeholder="Only required for private repos" value={gitToken} onChange={(event) => setGitToken(event.target.value)} />
              </Form.Item>
          <Space>
            <Button type="primary" onClick={handleGitAnalysis} loading={loading}>
              Analyze repository
            </Button>
            <Button onClick={() => window.open('https://reactjs.org/link/react-devtools', '_blank')}>Open React DevTools</Button>
          </Space>
        </Form>
      ),
    },
  ];

  const shellStyle: React.CSSProperties = {
    background: '#0d1117',
    borderRadius: 12,
    boxShadow: '0 20px 60px rgba(0,0,0,0.35)',
    padding: 0,
    display: 'flex',
    flexDirection: 'column',
    minHeight: '80vh',
  };

  return (
    <div style={{ padding: 24 }}>
      <div style={shellStyle}>
        <div
          style={{
            padding: '16px 24px',
            borderBottom: '1px solid #1f2530',
            background: '#151a24',
            color: '#f5f6fa',
            fontWeight: 600,
            fontSize: 16,
            letterSpacing: 0.5,
          }}
        >
          智能代码审查与架构分析平台
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#f7f8fa' }}>
          <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
            <div
              style={{
                width: explorerWidth,
                background: '#1e1e1e',
                color: '#d4d4d4',
                borderRight: '1px solid #000',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <div style={{ padding: '12px 16px', fontSize: 13, textTransform: 'uppercase', color: '#9cdcfe' }}>Explorer</div>
              <div style={{ flex: 1, overflow: 'auto', padding: '0 8px 16px' }}>
                {explorerSections.map((section) => (
                  <Card
                    key={section.title}
                    size="small"
                    style={{ background: '#252526', color: '#d4d4d4', marginBottom: 12 }}
                    styles={{ body: { padding: '8px 12px' } }}
                    title={<span style={{ color: '#9cdcfe', fontSize: 12 }}>{section.title}</span>}
                  >
                    <List
                      size="small"
                      dataSource={section.items}
                      renderItem={(item) => (
                        <List.Item style={{ color: '#d4d4d4' }}>
                          <Space>
                            <FolderOpenOutlined />
                            <span>{item}</span>
                          </Space>
                        </List.Item>
                      )}
                    />
                  </Card>
                ))}
              </div>
            </div>
            <div
              style={{ width: 6, cursor: 'col-resize', background: '#c4c4c4', opacity: 0.3 }}
              onMouseDown={startResizing('explorer')}
            />
            <div style={{ flex: 1, background: '#ffffff', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <div style={{ padding: '8px 16px', borderBottom: '1px solid #f0f0f0', display: 'flex', gap: 12 }}>
                <Badge status="processing" text="analysis.tsx" />
                <Badge status="default" text="baseline.json" />
              </div>
              <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
                <Card styles={{ body: { padding: 0 } }}>
                  <Tabs items={analysisTabs} />
                </Card>
                <Card style={{ marginTop: 16 }}>
                  <Space size="large">
                    <Space direction="vertical">
                      <Text type="secondary">Live status</Text>
                      <Space>
                        <BranchesOutlined /> Branch: main
                      </Space>
                      <Space>
                        <ThunderboltOutlined /> Sessions in queue: 2
                      </Space>
                    </Space>
                    <Space direction="vertical">
                      <Button icon={<PlayCircleOutlined />}>Start session</Button>
                      <Button icon={<PauseCircleOutlined />}>Pause session</Button>
                    </Space>
                  </Space>
      </Card>
                <div style={{ marginTop: 16 }}>
                  {loading && <Alert message="Analyzing your code..." type="info" showIcon style={{ marginBottom: 16 }} />}
                  {analysisResult && (
                    <Card>
                      <Tabs
                        items={[
                          { key: 'quality', label: <span><BarChartOutlined /> Quality</span>, children: renderQualityResults() },
                          { key: 'security', label: <span><SafetyOutlined /> Security</span>, children: renderSecurityResults() },
                          { key: 'architecture', label: <span><BugOutlined /> Architecture</span>, children: renderArchitectureResults() },
                        ]}
                      />
        </Card>
      )}
                </div>
                <div style={{ marginTop: 16 }}>
                  <CodeReviewWorkbench />
                </div>
              </div>
            </div>
            <div
              style={{ width: 6, cursor: 'col-resize', background: '#c4c4c4', opacity: 0.3 }}
              onMouseDown={startResizing('copilot')}
            />
            <div
              style={{
                width: copilotWidth,
                borderLeft: '1px solid #e4e6eb',
                background: '#f5f7fb',
                display: 'flex',
              }}
            >
              <AIChatPanel style={{ padding: 0, width: '100%' }} />
            </div>
          </div>
          <div
            style={{ height: 6, cursor: 'row-resize', background: '#c4c4c4', opacity: 0.4 }}
            onMouseDown={startResizing('bottom')}
          />
          <div style={{ height: bottomPanelHeight, background: '#0f141c', color: '#f5f5f5', padding: '0 16px' }}>
            <Tabs
              defaultActiveKey="terminal"
              type="card"
              items={[
                {
                  key: 'terminal',
                  label: (
                <span>
                      <ConsoleSqlOutlined /> Terminal
                </span>
                  ),
                  children: (
                    <pre style={{ margin: 0, padding: 16, minHeight: 140, color: '#9cdcfe' }}>
                      {terminalLines.join('\n')}
                    </pre>
                  ),
                },
                {
                  key: 'debug',
                  label: (
                <span>
                      <BugOutlined /> Debug console
                </span>
                  ),
                  children: (
                    <div style={{ padding: 16 }}>
                      <Paragraph style={{ color: '#f5f5f5' }}>No active breakpoints. Attach the runtime to stream traces here.</Paragraph>
                    </div>
                  ),
                },
              ]}
            />
          </div>
        </div>
        <div
          style={{
            textAlign: 'center',
            padding: 16,
            borderTop: '1px solid #1f2530',
            color: '#8c8c8c',
            fontSize: 12,
            background: '#151a24',
          }}
        >
          © 2025 智能代码审查与架构分析平台. 保留所有权利.
        </div>
      </div>
    </div>
  );
};

export default CodeAnalysis;

