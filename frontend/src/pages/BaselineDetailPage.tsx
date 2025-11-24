import React from 'react';
import {
  Card,
  Typography,
  Descriptions,
  Space,
  Tag,
  List,
  Button,
  Form,
  Input,
  message,
} from 'antd';
import { useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import type { Baseline } from './Baselines';

const { Title, Paragraph } = Typography;
const { TextArea } = Input;

const fallbackBaseline: Baseline = {
  id: 0,
  project_id: 0,
  name: 'Baseline',
  description: 'Fallback baseline',
  config: {},
  created_at: new Date().toISOString(),
  deviations: [],
};

const BaselineDetailPage: React.FC = () => {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const mode = searchParams.get('mode') || 'view';
  const [form] = Form.useForm();
  const baseline: Baseline = state?.baseline || fallbackBaseline;

  React.useEffect(() => {
    if (mode === 'edit') {
      form.setFieldsValue({
        name: baseline.name,
        description: baseline.description,
        config: JSON.stringify(baseline.config, null, 2),
      });
    }
  }, [baseline, form, mode]);

  const handleSave = (values: any) => {
    message.success('Baseline changes saved (local demo only)');
    navigate('/baselines');
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Card
        title={baseline.name}
        extra={
          <Space>
            <Button onClick={() => navigate('/baselines')}>Back to baselines</Button>
            <Button type="primary" onClick={() => navigate(`/baselines/${baseline.id}?mode=edit`, { state: { baseline } })}>
              Edit baseline
            </Button>
            <Button onClick={() => navigate(`/baselines/${baseline.id}?mode=deviations`, { state: { baseline } })}>
              View deviations
            </Button>
          </Space>
        }
      >
        <Descriptions bordered column={2} size="small">
          <Descriptions.Item label="Project ID">{baseline.project_id}</Descriptions.Item>
          <Descriptions.Item label="Created at">
            {new Date(baseline.created_at).toLocaleString()}
          </Descriptions.Item>
          <Descriptions.Item label="Description" span={2}>
            {baseline.description}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      {mode === 'edit' ? (
        <Card title="Edit baseline">
          <Form layout="vertical" form={form} onFinish={handleSave}>
            <Form.Item name="name" label="Baseline name" rules={[{ required: true }]}>
              <Input />
            </Form.Item>
            <Form.Item name="description" label="Description" rules={[{ required: true }]}>
              <TextArea rows={3} />
            </Form.Item>
            <Form.Item
              name="config"
              label="Config (JSON)"
              rules={[{ required: true, message: 'Provide the config JSON' }]}
            >
              <TextArea rows={6} />
            </Form.Item>
            <Button type="primary" htmlType="submit">
              Save changes
            </Button>
          </Form>
        </Card>
      ) : (
        <Card title="Configuration">
          <Paragraph>
            The baseline monitors the following metrics. Update the JSON with thresholds,
            tags, or SLA notes.
          </Paragraph>
          <pre style={{ background: '#0f172a', color: '#e2e8f0', padding: 16, borderRadius: 4 }}>
            {JSON.stringify(baseline.config, null, 2)}
          </pre>
        </Card>
      )}

      <Card title="Deviations">
        {baseline.deviations.length === 0 ? (
          <Paragraph>No deviations detected.</Paragraph>
        ) : (
          <List
            dataSource={baseline.deviations}
            renderItem={(deviation) => (
              <List.Item>
                <Space direction="vertical">
                  <Space>
                    <Tag color="red">{deviation.severity.toUpperCase()}</Tag>
                    <strong>{deviation.metric_name}</strong>
                  </Space>
                  <Paragraph style={{ marginBottom: 0 }}>
                    Deviation: {deviation.deviation_value}
                  </Paragraph>
                  <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    Detected at: {new Date(deviation.detected_at).toLocaleString()}
                  </Paragraph>
                </Space>
              </List.Item>
            )}
          />
        )}
      </Card>
    </Space>
  );
};

export default BaselineDetailPage;

