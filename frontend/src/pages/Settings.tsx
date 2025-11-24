import React from 'react';
import { Card, Typography, Form, Switch, Select, Button, Space, Input, Tabs, Alert, List, message } from 'antd';

const { Title, Paragraph } = Typography;
const { TextArea } = Input;

const Settings: React.FC = () => {
  const [generalForm] = Form.useForm();
  const [providerForm] = Form.useForm();

  const handleSave = () => {
    message.success('Settings saved for this demo session');
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card>
        <Title level={3}>Platform settings</Title>
        <Paragraph type="secondary">
          Configure feature flags, provider credentials, and notification routing. These
          values would normally be stored in a secure backend vault—here they are mocked for
          demonstration purposes.
        </Paragraph>
      </Card>

      <Card title="General preferences">
        <Form layout="vertical" form={generalForm} onFinish={handleSave}>
          <Form.Item label="Dark theme">
            <Switch />
          </Form.Item>
          <Form.Item label="Enable experimental features">
            <Switch />
          </Form.Item>
          <Form.Item label="Default insight density" name="insightDensity" initialValue="balanced">
            <Select
              options={[
                { value: 'balanced', label: 'Balanced' },
                { value: 'compact', label: 'Compact' },
                { value: 'verbose', label: 'Verbose' },
              ]}
            />
          </Form.Item>
          <Space>
            <Button onClick={() => generalForm.resetFields()}>Reset</Button>
            <Button type="primary" htmlType="submit">
              Save preferences
            </Button>
          </Space>
        </Form>
      </Card>

      <Card title="AI provider configuration">
        <Tabs
          items={[
            {
              key: 'openai',
              label: 'OpenAI',
              children: (
                <Form layout="vertical" form={providerForm} onFinish={handleSave}>
                  <Form.Item label="API base URL" name="openaiBase">
                    <Input placeholder="https://api.openai.com/v1" />
                  </Form.Item>
                  <Form.Item label="API key" name="openaiKey">
                    <Input.Password placeholder="sk-..." />
                  </Form.Item>
                  <Button type="primary" htmlType="submit">
                    Save OpenAI settings
                  </Button>
                </Form>
              ),
            },
            {
              key: 'azure',
              label: 'Azure OpenAI',
              children: (
                <Form layout="vertical" onFinish={handleSave}>
                  <Form.Item label="Resource name">
                    <Input placeholder="my-azure-openai" />
                  </Form.Item>
                  <Form.Item label="Deployment ID">
                    <Input placeholder="gpt-4o-mini" />
                  </Form.Item>
                  <Button type="primary" htmlType="submit">
                    Save Azure settings
                  </Button>
                </Form>
              ),
            },
          ]}
        />
      </Card>

      <Card title="Notification routing">
        <Form layout="vertical" onFinish={handleSave}>
          <Form.Item label="Webhook URL">
            <Input placeholder="https://hooks.slack.com/..." />
          </Form.Item>
          <Form.Item label="Escalation emails">
            <Select
              mode="tags"
              placeholder="security@example.com"
              options={[]}
            />
          </Form.Item>
          <Form.Item label="Alert template">
            <TextArea rows={4} placeholder="[Project] [Severity] message body" />
          </Form.Item>
          <Button type="primary" htmlType="submit">
            Save routing
          </Button>
        </Form>
      </Card>

      <Card title="Example demonstrations">
        <List
          dataSource={[
            'Switch feature flags to preview the AI chat dock across multiple tenants.',
            'Configure Azure OpenAI and OpenAI simultaneously, then pick the provider in the chat panel.',
            'Push webhook updates to Slack and email to keep architecture stakeholders informed.',
          ]}
          renderItem={(item) => (
            <List.Item>
              <Alert message={item} type="info" showIcon />
            </List.Item>
          )}
        />
      </Card>
    </Space>
  );
};

export default Settings;
