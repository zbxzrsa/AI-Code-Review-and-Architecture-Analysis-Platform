import React, { useState } from 'react';
import { Card, Typography, Select, Input, Button, List, Space, Tag } from 'antd';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

type Provider = 'OpenAI' | 'Azure OpenAI' | 'Anthropic' | 'GLM-4.6';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  provider: Provider;
}

interface AIChatPanelProps {
  style?: React.CSSProperties;
}

const providerColors: Record<Provider, string> = {
  OpenAI: 'blue',
  'Azure OpenAI': 'geekblue',
  Anthropic: 'purple',
  'GLM-4.6': 'green',
};

const quickActions = [
  { label: 'Summarize issues', prompt: 'Summarize the top 3 issues blocking release readiness.' },
  { label: 'Suggest tests', prompt: 'Suggest integration tests for the latest session.' },
  { label: 'Explain diff', prompt: 'Explain the main changes in analysis.tsx.' },
];

const AIChatPanel: React.FC<AIChatPanelProps> = ({ style }) => {
  const [provider, setProvider] = useState<Provider>('OpenAI');
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'msg-1',
      role: 'assistant',
      provider: 'OpenAI',
      content: 'Welcome! I can review logs, summarize diffs, and draft remediation steps.',
    },
    {
      id: 'msg-2',
      role: 'assistant',
      provider: 'OpenAI',
      content: '```ts\nconst guardrail = (score: number) => score >= 85 ? "ship" : "hold";\n```',
    },
  ]);

  const pushAssistantReply = (prompt: string) => {
    const response: ChatMessage = {
      id: `msg-${Date.now() + 1}`,
      role: 'assistant',
      provider,
      content: `**${provider}** summary:\n\n- Detected trend for: \`${prompt}\`\n- Confidence: 0.91\n- Use \`ai fix --scope core-service\` to open a remediation checklist.`,
    };
    setMessages((prev) => [...prev, response]);
  };

  const handleSend = (override?: string) => {
    const text = (override ?? message).trim();
    if (!text) {
      return;
    }

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      provider,
    };
    setMessages((prev) => [...prev, userMessage]);
    setMessage('');
    setTimeout(() => pushAssistantReply(text), 400);
  };

  return (
    <Card
      style={{ display: 'flex', flexDirection: 'column', width: '100%', ...style }}
      styles={{ body: { display: 'flex', flexDirection: 'column', flex: 1, padding: 16 } }}
      title={
        <Space>
          <Typography.Text strong>AI Copilot</Typography.Text>
          <Select<Provider>
            value={provider}
            onChange={setProvider}
            size="small"
            options={[
              { value: 'OpenAI', label: 'OpenAI' },
              { value: 'Azure OpenAI', label: 'Azure OpenAI' },
              { value: 'Anthropic', label: 'Anthropic' },
              { value: 'GLM-4.6', label: 'GLM-4.6 (Cloud)' },
            ]}
          />
        </Space>
      }
    >
      <Space wrap style={{ marginBottom: 12 }}>
        {quickActions.map((action) => (
          <Button key={action.label} size="small" onClick={() => handleSend(action.prompt)}>
            {action.label}
          </Button>
        ))}
      </Space>
      <List
        size="small"
        dataSource={messages}
        style={{ flex: 1, overflow: 'auto', marginBottom: 12 }}
        renderItem={(item) => (
          <List.Item>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Tag color={providerColors[item.provider]}>{item.provider}</Tag>
                <Typography.Text strong>{item.role === 'user' ? 'You' : 'Assistant'}</Typography.Text>
              </Space>
              <ReactMarkdown
                remarkPlugins={[remarkGfm as any]}
                components={{
                  code(props: any) {
                    const { inline, className, children, ...rest } = props;
                    const match = /language-(\w+)/.exec(className || '');
                    if (!inline) {
                      return (
                        <SyntaxHighlighter
                          style={oneDark as any}
                          language={match ? match[1] : 'text'}
                          PreTag="div"
                          {...rest}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      );
                    }
                    return (
                      <code className={className} {...rest}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {item.content}
              </ReactMarkdown>
            </Space>
          </List.Item>
        )}
      />
      <Input.TextArea
        rows={3}
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        placeholder="Ask for a remediation plan or diff summary..."
      />
      <Button type="primary" onClick={() => handleSend()} style={{ marginTop: 8 }}>
        Send
      </Button>
    </Card>
  );
};

export default AIChatPanel;

