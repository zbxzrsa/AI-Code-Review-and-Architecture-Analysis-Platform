import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Space, Tooltip, Modal, Form, Input, Select, message, Progress, Statistic, Row, Col, List } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined, EyeOutlined, AlertOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useNavigate } from 'react-router-dom';

const { TextArea } = Input;

export interface BaselineDeviation {
  id: number;
  baseline_id: number;
  metric_name: string;
  deviation_value: number;
  severity: string;
  detected_at: string;
}

export interface Baseline {
  id: number;
  project_id: number;
  name: string;
  description: string;
  config: Record<string, any>;
  created_at: string;
  deviations: BaselineDeviation[];
}

interface BaselineStatus {
  baseline_id: number;
  status: string;
  total_deviations: number;
  deviation_counts: Record<string, number>;
  last_check: string;
}

const Baselines: React.FC = () => {
  const [baselines, setBaselines] = useState<Baseline[]>([]);
  const [baselineStatuses, setBaselineStatuses] = useState<Record<number, BaselineStatus>>({});
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();
  const navigate = useNavigate();

  const statusColors: Record<string, string> = {
    'healthy': 'green',
    'attention': 'orange',
    'warning': 'red',
    'critical': 'purple'
  };

  const severityColors: Record<string, string> = {
    'low': 'blue',
    'medium': 'orange',
    'high': 'red',
    'critical': 'purple'
  };

  const columns: ColumnsType<Baseline> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Baseline Name',
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: 'Project ID',
      dataIndex: 'project_id',
      key: 'project_id',
      width: 100,
    },
    {
      title: 'Status',
      key: 'status',
      width: 120,
      render: (_, record) => {
        const status = baselineStatuses[record.id];
        if (!status) return <Tag>Unknown</Tag>;
        
        return (
          <Tooltip title={`${status.total_deviations} deviations`}>
            <Tag 
              color={statusColors[status.status]} 
              icon={status.status === 'healthy' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
            >
              {status.status.toUpperCase()}
            </Tag>
          </Tooltip>
        );
      },
    },
    {
      title: 'Deviation Stats',
      key: 'deviations',
      width: 200,
      render: (_, record) => {
        const status = baselineStatuses[record.id];
        if (!status || status.total_deviations === 0) {
          return <span style={{ color: '#52c41a' }}>No deviations</span>;
        }
        
        return (
          <Space size="small">
            {Object.entries(status.deviation_counts).map(([severity, count]) => (
              count > 0 && (
                <Tag key={severity} color={severityColors[severity]}>
                  {severity}: {count}
                </Tag>
              )
            ))}
          </Space>
        );
      },
    },
    {
      title: 'Created At',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      render: (_, record) => (
        <Space>
          <Tooltip title="View Details">
            <Button 
              type="link" 
              icon={<EyeOutlined />} 
              onClick={() => handleViewBaseline(record)}
            />
          </Tooltip>
          <Tooltip title="Edit Baseline">
            <Button 
              type="link" 
              icon={<EditOutlined />} 
              onClick={() => handleViewBaseline(record, 'edit')}
            />
          </Tooltip>
          <Tooltip title="View Deviations">
            <Button 
              type="link" 
              icon={<AlertOutlined />} 
              onClick={() => handleViewDeviations(record)}
            />
          </Tooltip>
          <Tooltip title="Delete Baseline">
            <Button 
              type="link" 
              danger 
              icon={<DeleteOutlined />} 
              onClick={() => handleDeleteBaseline(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const fetchBaselines = async () => {
    setLoading(true);
    try {
      // TODO: Replace with actual API call
      const mockData: Baseline[] = [
        {
          id: 1,
          project_id: 1,
          name: "Quality Baseline v1.0",
          description: "Initial quality metrics baseline for e-commerce platform",
          config: {
            metrics: {
              code_coverage: { min: 80.0, target: 90.0 },
              cyclomatic_complexity: { max: 10.0, target: 5.0 }
            }
          },
          created_at: "2024-01-15T09:00:00Z",
          deviations: []
        },
        {
          id: 2,
          project_id: 1,
          name: "Security Baseline v1.0",
          description: "Security metrics baseline including vulnerability thresholds",
          config: {
            metrics: {
              security_score: { min: 85.0, target: 95.0 },
              vulnerability_count: { max: 0, target: 0 }
            }
          },
          created_at: "2024-01-15T09:30:00Z",
          deviations: []
        }
      ];
      
      setBaselines(mockData);
      
      // Fetch status for each baseline
      const statuses: Record<number, BaselineStatus> = {};
      for (const baseline of mockData) {
        // TODO: Replace with actual API call
        statuses[baseline.id] = {
          baseline_id: baseline.id,
          status: baseline.id === 1 ? 'warning' : 'healthy',
          total_deviations: baseline.id === 1 ? 2 : 0,
          deviation_counts: baseline.id === 1 ? { medium: 1, high: 1 } : {},
          last_check: new Date().toISOString()
        };
      }
      setBaselineStatuses(statuses);
      
    } catch (error) {
      message.error('Failed to fetch baselines');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateBaseline = async (values: any) => {
    try {
      const parsedConfig = JSON.parse(values.config);
      const newBaseline: Baseline = {
        id: Date.now(),
        project_id: Number(values.project_id),
        name: values.name,
        description: values.description,
        config: parsedConfig,
        created_at: new Date().toISOString(),
        deviations: [],
      };
      setBaselines((prev) => [newBaseline, ...prev]);
      setBaselineStatuses((prev) => ({
        ...prev,
        [newBaseline.id]: {
          baseline_id: newBaseline.id,
          status: 'healthy',
          total_deviations: 0,
          deviation_counts: {},
          last_check: new Date().toISOString(),
        },
      }));
      message.success('Baseline created successfully');
      setCreateModalVisible(false);
      form.resetFields();
    } catch (error) {
      message.error('Failed to create baseline');
    }
  };

  const handleViewBaseline = (baseline: Baseline, mode: string = 'view') => {
    navigate(`/baselines/${baseline.id}?mode=${mode}`, { state: { baseline } });
  };

  const handleViewDeviations = (baseline: Baseline) => {
    navigate(`/baselines/${baseline.id}?mode=deviations`, { state: { baseline } });
  };

  const handleDeleteBaseline = (baselineId: number) => {
    Modal.confirm({
      title: 'Confirm Delete',
      content: 'Are you sure to delete this baseline? This action cannot be undone.',
      onOk: async () => {
        try {
          setBaselines((prev) => prev.filter((baseline) => baseline.id !== baselineId));
          setBaselineStatuses((prev) => {
            const next = { ...prev };
            delete next[baselineId];
            return next;
          });
          message.success('Baseline deleted successfully');
        } catch (error) {
          message.error('Failed to delete baseline');
        }
      },
    });
  };

  const getOverallHealth = () => {
    const statuses = Object.values(baselineStatuses);
    if (statuses.length === 0) return { status: 'unknown', percentage: 0 };
    
    const healthyCount = statuses.filter(s => s.status === 'healthy').length;
    const percentage = (healthyCount / statuses.length) * 100;
    
    let status = 'healthy';
    if (percentage < 50) status = 'critical';
    else if (percentage < 80) status = 'warning';
    else if (percentage < 100) status = 'attention';
    
    return { status, percentage };
  };

  const getTotalDeviations = () => {
    return Object.values(baselineStatuses).reduce((sum, status) => sum + status.total_deviations, 0);
  };

  useEffect(() => {
    fetchBaselines();
  }, []);

  const overallHealth = getOverallHealth();
  const totalDeviations = getTotalDeviations();

  return (
    <div style={{ padding: '24px' }}>
      {/* Overview Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Baselines"
              value={baselines.length}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Health"
              value={overallHealth.percentage}
              precision={1}
              suffix="%"
              valueStyle={{ color: statusColors[overallHealth.status] }}
            />
            <Progress 
              percent={overallHealth.percentage} 
              strokeColor={statusColors[overallHealth.status]}
              showInfo={false}
              size="small"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Deviations"
              value={totalDeviations}
              prefix={<AlertOutlined />}
              valueStyle={{ color: totalDeviations > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Active Baselines"
              value={Object.values(baselineStatuses).filter(s => s.status !== 'healthy').length}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Baselines Table */}
      <Card 
        title="Baseline Management" 
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => setCreateModalVisible(true)}
          >
            Create Baseline
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={baselines}
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `Total ${total} records`,
          }}
        />
      </Card>

      {/* Create Modal */}
      <Modal
        title="Create Baseline"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateBaseline}
        >
          <Form.Item
            name="project_id"
            label="Project ID"
            rules={[{ required: true, message: 'Please enter Project ID' }]}
          >
            <Input type="number" placeholder="Please enter Project ID" />
          </Form.Item>
          <Form.Item
            name="name"
            label="Baseline Name"
            rules={[{ required: true, message: 'Please enter Baseline Name' }]}
          >
            <Input placeholder="Please enter Baseline Name" />
          </Form.Item>
          <Form.Item
            name="description"
            label="Description"
            rules={[{ required: true, message: 'Please enter Description' }]}
          >
            <TextArea rows={3} placeholder="Please enter baseline description" />
          </Form.Item>
          <Form.Item
            name="config"
            label="Config (JSON)"
            rules={[{ required: true, message: 'Please enter Config' }]}
          >
            <TextArea 
              rows={8} 
              placeholder='{"metrics": {"code_coverage": {"min": 80.0, "target": 90.0}}}'
            />
          </Form.Item>
        </Form>
      </Modal>

      <Card title="Example demonstrations" style={{ marginTop: 16 }}>
        <List
          dataSource={[
            'Promote the performance baseline before the holiday release to detect regressions early.',
            'Compare the security baseline against the latest penetration test findings.',
            'Export the JSON config to share with another region or environment.',
          ]}
          renderItem={(item) => (
            <List.Item>
              <span>{item}</span>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default Baselines;