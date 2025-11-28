import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Switch,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Space,
  Tag,
  Tooltip,
  message,
  Popconfirm,
  Badge,
  Alert,
  Row,
  Col,
  Statistic,
  Progress,
  Timeline,
  Descriptions,
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  HistoryOutlined,
  SettingOutlined,
  DashboardOutlined,
  AlertOutlined,
} from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;

interface FeatureFlag {
  id: string;
  key: string;
  name: string;
  description: string;
  type: 'boolean' | 'string' | 'number' | 'json';
  enabled: boolean;
  value: any;
  rollout: {
    percentage: number;
    strategy: 'gradual' | 'immediate' | 'scheduled';
  };
  kill_switch: {
    enabled: boolean;
    reason?: string;
    triggered_by?: string;
    triggered_at?: string;
  };
  metadata: {
    created_by: string;
    created_at: string;
    updated_by: string;
    updated_at: string;
    tags: string[];
  };
  stats?: {
    evaluation_count: number;
    enabled_count: number;
    disabled_count: number;
    cache_hit_rate: number;
    avg_evaluation_time: number;
  };
}

interface AuditEvent {
  id: string;
  timestamp: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  old_value?: any;
  new_value?: any;
  metadata: any;
}

const FeatureFlagDashboard: React.FC = () => {
  const [flags, setFlags] = useState<FeatureFlag[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingFlag, setEditingFlag] = useState<FeatureFlag | null>(null);
  const [form] = Form.useForm();
  const [auditModalVisible, setAuditModalVisible] = useState(false);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [emergencyModalVisible, setEmergencyModalVisible] = useState(false);
  const [selectedFlag, setSelectedFlag] = useState<FeatureFlag | null>(null);
  const [stats, setStats] = useState({
    totalFlags: 0,
    enabledFlags: 0,
    killSwitchActive: 0,
    avgEvaluationTime: 0,
  });

  // Load feature flags
  const loadFlags = async () => {
    setLoading(true);
    try {
      // This would call your API
      const response = await fetch('/api/feature-flags');
      const data = await response.json();
      setFlags(data.flags || []);
      setStats(data.stats || stats);
    } catch (error) {
      message.error('Failed to load feature flags');
      console.error('Error loading flags:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load audit events
  const loadAuditEvents = async (flagKey?: string) => {
    try {
      const url = flagKey ? `/api/audit-events?flag_key=${flagKey}` : '/api/audit-events';
      const response = await fetch(url);
      const data = await response.json();
      setAuditEvents(data.events || []);
    } catch (error) {
      message.error('Failed to load audit events');
      console.error('Error loading audit events:', error);
    }
  };

  useEffect(() => {
    loadFlags();
    // Set up real-time updates
    const interval = setInterval(loadFlags, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Toggle flag
  const toggleFlag = async (flag: FeatureFlag, enabled: boolean) => {
    try {
      await fetch(`/api/feature-flags/${flag.key}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      message.success(`Flag ${flag.key} ${enabled ? 'enabled' : 'disabled'}`);
      loadFlags();
    } catch (error) {
      message.error('Failed to toggle flag');
      console.error('Error toggling flag:', error);
    }
  };

  // Activate kill switch
  const activateKillSwitch = async (flag: FeatureFlag, reason: string) => {
    try {
      await fetch(`/api/feature-flags/${flag.key}/kill-switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason, action: 'activate' }),
      });
      message.success(`Emergency kill switch activated for ${flag.key}`);
      setEmergencyModalVisible(false);
      loadFlags();
    } catch (error) {
      message.error('Failed to activate kill switch');
      console.error('Error activating kill switch:', error);
    }
  };

  // Deactivate kill switch
  const deactivateKillSwitch = async (flag: FeatureFlag) => {
    try {
      await fetch(`/api/feature-flags/${flag.key}/kill-switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'deactivate' }),
      });
      message.success(`Kill switch deactivated for ${flag.key}`);
      loadFlags();
    } catch (error) {
      message.error('Failed to deactivate kill switch');
      console.error('Error deactiving kill switch:', error);
    }
  };

  // Save flag
  const saveFlag = async (values: any) => {
    try {
      const method = editingFlag ? 'PUT' : 'POST';
      const url = editingFlag ? `/api/feature-flags/${editingFlag.key}` : '/api/feature-flags';

      await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });

      message.success(`Flag ${editingFlag ? 'updated' : 'created'} successfully`);
      setModalVisible(false);
      setEditingFlag(null);
      form.resetFields();
      loadFlags();
    } catch (error) {
      message.error(`Failed to ${editingFlag ? 'update' : 'create'} flag`);
      console.error('Error saving flag:', error);
    }
  };

  // Delete flag
  const deleteFlag = async (flag: FeatureFlag) => {
    try {
      await fetch(`/api/feature-flags/${flag.key}`, {
        method: 'DELETE',
      });
      message.success(`Flag ${flag.key} deleted successfully`);
      loadFlags();
    } catch (error) {
      message.error('Failed to delete flag');
      console.error('Error deleting flag:', error);
    }
  };

  // Table columns
  const columns = [
    {
      title: 'Key',
      dataIndex: 'key',
      key: 'key',
      render: (key: string, record: FeatureFlag) => (
        <Space>
          <span style={{ fontFamily: 'monospace' }}>{key}</span>
          {record.kill_switch?.enabled && <Badge status="error" text="KILL SWITCH" />}
        </Space>
      ),
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Status',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: FeatureFlag) => (
        <Space>
          <Switch
            checked={enabled}
            onChange={checked => toggleFlag(record, checked)}
            disabled={record.kill_switch?.enabled}
            checkedChildren="ON"
            unCheckedChildren="OFF"
          />
          {record.kill_switch?.enabled ? (
            <Tag color="red" icon={<ThunderboltOutlined />}>
              Emergency
            </Tag>
          ) : enabled ? (
            <Tag color="green" icon={<CheckCircleOutlined />}>
              Active
            </Tag>
          ) : (
            <Tag color="default" icon={<CloseCircleOutlined />}>
              Inactive
            </Tag>
          )}
        </Space>
      ),
    },
    {
      title: 'Rollout',
      key: 'rollout',
      render: (record: FeatureFlag) => (
        <Space direction="vertical" size="small">
          <Progress
            percent={record.rollout.percentage}
            size="small"
            status={record.rollout.percentage === 100 ? 'success' : 'active'}
          />
          <Tag color="blue">{record.rollout.strategy}</Tag>
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag>{type.toUpperCase()}</Tag>,
    },
    {
      title: 'Updated',
      dataIndex: ['metadata', 'updated_at'],
      key: 'updated_at',
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (record: FeatureFlag) => (
        <Space>
          <Tooltip title="Edit">
            <Button
              icon={<EditOutlined />}
              size="small"
              onClick={() => {
                setEditingFlag(record);
                form.setFieldsValue(record);
                setModalVisible(true);
              }}
            />
          </Tooltip>

          <Tooltip title="View History">
            <Button
              icon={<HistoryOutlined />}
              size="small"
              onClick={() => {
                setSelectedFlag(record);
                loadAuditEvents(record.key);
                setAuditModalVisible(true);
              }}
            />
          </Tooltip>

          {record.kill_switch?.enabled ? (
            <Tooltip title="Deactivate Kill Switch">
              <Popconfirm
                title="Deactivate kill switch?"
                description="This will re-enable normal flag operation"
                onConfirm={() => deactivateKillSwitch(record)}
                okText="Deactivate"
                cancelText="Cancel"
              >
                <Button icon={<CheckCircleOutlined />} size="small" type="primary" ghost />
              </Popconfirm>
            </Tooltip>
          ) : (
            <Tooltip title="Emergency Kill Switch">
              <Popconfirm
                title="Activate emergency kill switch?"
                description="This will immediately disable the flag for all users"
                onConfirm={() => {
                  setSelectedFlag(record);
                  setEmergencyModalVisible(true);
                }}
                okText="Activate"
                cancelText="Cancel"
                okType="danger"
              >
                <Button icon={<ThunderboltOutlined />} size="small" danger />
              </Popconfirm>
            </Tooltip>
          )}

          <Tooltip title="Delete">
            <Popconfirm
              title="Delete this flag?"
              description="This action cannot be undone"
              onConfirm={() => deleteFlag(record)}
              okText="Delete"
              cancelText="Cancel"
              okType="danger"
            >
              <Button icon={<DeleteOutlined />} size="small" danger />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* Header Stats */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Flags"
              value={stats.totalFlags}
              prefix={<DashboardOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Enabled Flags"
              value={stats.enabledFlags}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Kill Switches Active"
              value={stats.killSwitchActive}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg Evaluation Time"
              value={stats.avgEvaluationTime}
              suffix="ms"
              prefix={<SettingOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Emergency Alert */}
      {stats.killSwitchActive > 0 && (
        <Alert
          message="Emergency Kill Switches Active"
          description={`${stats.killSwitchActive} feature flag(s) are currently under emergency control. Review and resolve as soon as possible.`}
          type="error"
          showIcon
          style={{ marginBottom: '24px' }}
          action={
            <Button size="small" danger>
              View Details
            </Button>
          }
        />
      )}

      {/* Main Table */}
      <Card
        title="Feature Flags"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingFlag(null);
              form.resetFields();
              setModalVisible(true);
            }}
          >
            Create Flag
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={flags}
          rowKey="key"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
          }}
        />
      </Card>

      {/* Create/Edit Modal */}
      <Modal
        title={editingFlag ? 'Edit Feature Flag' : 'Create Feature Flag'}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          setEditingFlag(null);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={saveFlag}>
          <Form.Item
            name="key"
            label="Flag Key"
            rules={[
              { required: true, message: 'Please enter flag key' },
              { pattern: /^[a-z_][a-z0-9_]*$/, message: 'Invalid flag key format' },
            ]}
          >
            <Input placeholder="e.g., new_dashboard_enabled" />
          </Form.Item>

          <Form.Item
            name="name"
            label="Display Name"
            rules={[{ required: true, message: 'Please enter display name' }]}
          >
            <Input placeholder="e.g., New Dashboard" />
          </Form.Item>

          <Form.Item name="description" label="Description">
            <TextArea rows={3} placeholder="Describe what this flag controls" />
          </Form.Item>

          <Form.Item
            name="type"
            label="Type"
            rules={[{ required: true, message: 'Please select flag type' }]}
          >
            <Select placeholder="Select flag type">
              <Option value="boolean">Boolean</Option>
              <Option value="string">String</Option>
              <Option value="number">Number</Option>
              <Option value="json">JSON</Option>
            </Select>
          </Form.Item>

          <Form.Item name="value" label="Default Value">
            <Input placeholder="Default value when enabled" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name={['rollout', 'percentage']}
                label="Rollout Percentage"
                initialValue={100}
              >
                <InputNumber min={0} max={100} formatter={value => `${value}%`} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name={['rollout', 'strategy']}
                label="Rollout Strategy"
                initialValue="immediate"
              >
                <Select>
                  <Option value="immediate">Immediate</Option>
                  <Option value="gradual">Gradual</Option>
                  <Option value="scheduled">Scheduled</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                {editingFlag ? 'Update' : 'Create'}
              </Button>
              <Button onClick={() => setModalVisible(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Emergency Kill Switch Modal */}
      <Modal
        title="🚨 Emergency Kill Switch"
        open={emergencyModalVisible}
        onCancel={() => setEmergencyModalVisible(false)}
        footer={null}
        width={500}
      >
        <Alert
          message="Emergency Action"
          description="Activating the kill switch will immediately disable this feature for all users. This should only be used in emergency situations."
          type="error"
          showIcon
          style={{ marginBottom: '16px' }}
        />

        {selectedFlag && (
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label="Flag Key">{selectedFlag.key}</Descriptions.Item>
            <Descriptions.Item label="Name">{selectedFlag.name}</Descriptions.Item>
            <Descriptions.Item label="Current Status">
              {selectedFlag.enabled ? (
                <Tag color="green">Enabled</Tag>
              ) : (
                <Tag color="default">Disabled</Tag>
              )}
            </Descriptions.Item>
          </Descriptions>
        )}

        <Form
          onFinish={values => {
            if (selectedFlag) {
              activateKillSwitch(selectedFlag, values.reason);
            }
          }}
          style={{ marginTop: '16px' }}
        >
          <Form.Item
            name="reason"
            label="Reason for Emergency"
            rules={[{ required: true, message: 'Please provide a reason' }]}
          >
            <TextArea rows={3} placeholder="Explain why this emergency action is necessary..." />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" danger htmlType="submit">
                Activate Kill Switch
              </Button>
              <Button onClick={() => setEmergencyModalVisible(false)}>Cancel</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Audit History Modal */}
      <Modal
        title="Audit History"
        open={auditModalVisible}
        onCancel={() => setAuditModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setAuditModalVisible(false)}>
            Close
          </Button>,
        ]}
        width={800}
      >
        <Timeline>
          {auditEvents.map(event => (
            <Timeline.Item
              key={event.id}
              color={
                event.action.includes('delete')
                  ? 'red'
                  : event.action.includes('create')
                    ? 'green'
                    : 'blue'
              }
            >
              <div>
                <strong>{event.action.replace('_', ' ').toUpperCase()}</strong>
                <br />
                {event.resource_id} • {event.user_id}
                <br />
                <small>{new Date(event.timestamp).toLocaleString()}</small>
              </div>
            </Timeline.Item>
          ))}
        </Timeline>
      </Modal>
    </div>
  );
};

export default FeatureFlagDashboard;
