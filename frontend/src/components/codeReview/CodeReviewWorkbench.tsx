import React, { useState } from 'react';
import { Card, Typography, List, Button, Space, Tag } from 'antd';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';

interface Suggestion {
  id: string;
  title: string;
  severity: 'info' | 'warning' | 'critical';
  description: string;
  apply: (code: string) => string;
}

const initialCode = `export async function fetchOrders(client) {
  const response = await client.get('/orders');
  const data = await response.json();
  return data.items.map((item) => ({
    id: item.id,
    total: item.total,
  }));
}
`;

const suggestions: Suggestion[] = [
  {
    id: 'sg-1',
    title: 'Add retry strategy',
    severity: 'warning',
    description: 'Transient network failures should be retried with exponential backoff.',
    apply: (code: string) =>
      code.replace(
        'const response = await client.get(\'/orders\');',
        `let response;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    response = await client.get('/orders');
    if (response.ok) break;
    await new Promise((resolve) => setTimeout(resolve, 100 * (attempt + 1)));
  }
  if (!response?.ok) {
    throw new Error('Unable to fetch orders after retries');
  }`
      ),
  },
  {
    id: 'sg-2',
    title: 'Validate payload shape',
    severity: 'info',
    description: 'Ensure the API response structure matches expectations before mapping.',
    apply: (code: string) =>
      code.replace(
        'const data = await response.json();',
        `const data = await response.json();
  if (!Array.isArray(data?.items)) {
    throw new Error('Invalid response format');
  }`
      ),
  },
  {
    id: 'sg-3',
    title: 'Mask sensitive fields',
    severity: 'critical',
    description: 'Remove PII before returning the payload to downstream services.',
    apply: (code: string) =>
      code.replace(
        'return data.items.map((item) => ({',
        `return data.items.map((item) => ({
    customerEmail: undefined,`
      ),
  },
];

const severityColor: Record<Suggestion['severity'], string> = {
  info: 'blue',
  warning: 'orange',
  critical: 'red',
};

const CodeReviewWorkbench: React.FC = () => {
  const [code, setCode] = useState(initialCode);

  const handleApply = (suggestion: Suggestion) => {
    setCode((current) => suggestion.apply(current));
  };

  return (
    <Card title="Code review workbench">
      <Typography.Paragraph type="secondary">
        Edit the snippet, apply AI suggestions, and capture the rationale before exporting
        comments to GitHub or your ticketing system.
      </Typography.Paragraph>
      <CodeMirror
        value={code}
        height="320px"
        theme={oneDark}
        extensions={[javascript()]}
        onChange={(value) => setCode(value)}
      />
      <Typography.Title level={5} style={{ marginTop: 24 }}>
        AI suggestions
      </Typography.Title>
      <List
        dataSource={suggestions}
        renderItem={(suggestion) => (
          <List.Item
            actions={[
              <Button key={suggestion.id} onClick={() => handleApply(suggestion)}>
                Apply
              </Button>,
            ]}
          >
            <List.Item.Meta
              title={
                <Space>
                  <Tag color={severityColor[suggestion.severity]}>{suggestion.severity}</Tag>
                  <Typography.Text strong>{suggestion.title}</Typography.Text>
                </Space>
              }
              description={suggestion.description}
            />
          </List.Item>
        )}
      />
    </Card>
  );
};

export default CodeReviewWorkbench;

