import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Space, Tooltip, Select, DatePicker, message, Tabs, Timeline, List } from 'antd';
import { HistoryOutlined, SwapOutlined, FileTextOutlined, BranchesOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { Dayjs } from 'dayjs';
import { useNavigate } from 'react-router-dom';

const { RangePicker } = DatePicker;


interface FileVersion {
  id: number;
  project_id: number;
  file_path: string;
  sha256: string;
  created_at: string;
}

interface VersionComparison {
  file_path: string;
  old_version?: FileVersion;
  new_version?: FileVersion;
  status: string;
}

const Versions: React.FC = () => {
  const navigate = useNavigate();
  const [versions, setVersions] = useState<FileVersion[]>([]);
  const [comparisons, setComparisons] = useState<VersionComparison[]>([]);
  const [projects, setProjects] = useState<any[]>([]);
  const [selectedProject, setSelectedProject] = useState<number | undefined>();
  const [selectedFile, setSelectedFile] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('versions');

  const statusColors: Record<string, string> = {
    'added': 'green',
    'modified': 'orange',
    'deleted': 'red',
    'unchanged': 'gray'
  };

  const versionColumns: ColumnsType<FileVersion> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'File Path',
      dataIndex: 'file_path',
      key: 'file_path',
      ellipsis: true,
    },
    {
      title: 'SHA256',
      dataIndex: 'sha256',
      key: 'sha256',
      width: 200,
      render: (sha256: string) => (
        <Tooltip title={sha256}>
          <code>{sha256.substring(0, 16)}...</code>
        </Tooltip>
      ),
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
      width: 150,
      render: (_, record) => (
        <Space>
          <Tooltip title="View History">
            <Button 
              type="link" 
              icon={<HistoryOutlined />} 
              onClick={() => handleViewHistory(record.file_path)}
            />
          </Tooltip>
          <Tooltip title="File Details">
            <Button 
              type="link" 
              icon={<FileTextOutlined />} 
              onClick={() => handleViewFile(record)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const comparisonColumns: ColumnsType<VersionComparison> = [
    {
      title: 'File Path',
      dataIndex: 'file_path',
      key: 'file_path',
      ellipsis: true,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={statusColors[status] || 'default'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Old Version',
      key: 'old_version',
      width: 200,
      render: (_, record) => record.old_version ? (
        <Tooltip title={record.old_version.sha256}>
          <code>{record.old_version.sha256.substring(0, 12)}...</code>
        </Tooltip>
      ) : '-',
    },
    {
      title: 'New Version',
      key: 'new_version',
      width: 200,
      render: (_, record) => record.new_version ? (
        <Tooltip title={record.new_version.sha256}>
          <code>{record.new_version.sha256.substring(0, 12)}...</code>
        </Tooltip>
      ) : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_, record) => (
        <Tooltip title="View Diff">
          <Button
            type="link"
            icon={<SwapOutlined />}
            onClick={() => handleViewDiff(record)}
          />
        </Tooltip>
      ),
    },
  ];

  const fetchVersions = async () => {
    setLoading(true);
    try {
      // TODO: Replace with actual API call
      const mockData: FileVersion[] = [
        {
          id: 1,
          project_id: 1,
          file_path: "src/main.py",
          sha256: "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
          created_at: "2024-01-15T10:00:00Z"
        },
        {
          id: 2,
          project_id: 1,
          file_path: "src/utils.py",
          sha256: "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
          created_at: "2024-01-15T10:05:00Z"
        },
        {
          id: 3,
          project_id: 1,
          file_path: "src/main.py",
          sha256: "c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678",
          created_at: "2024-01-20T14:00:00Z"
        }
      ];
      
      let filteredData = mockData;
      if (selectedProject) {
        filteredData = filteredData.filter(v => v.project_id === selectedProject);
      }
      if (selectedFile) {
        filteredData = filteredData.filter(v => v.file_path.includes(selectedFile));
      }
      
      setVersions(filteredData);
    } catch (error) {
      message.error('Failed to fetch versions');
    } finally {
      setLoading(false);
    }
  };

  const handleCompareVersions = async (dateRange: [Dayjs, Dayjs] | null) => {
    if (!selectedProject || !dateRange) {
      message.warning('Please select project and time range');
      return;
    }

    setCompareLoading(true);
    try {
      // TODO: Replace with actual API call
      const mockComparisons: VersionComparison[] = [
        {
          file_path: "src/main.py",
          old_version: {
            id: 1,
            project_id: 1,
            file_path: "src/main.py",
            sha256: "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
            created_at: "2024-01-15T10:00:00Z"
          },
          new_version: {
            id: 3,
            project_id: 1,
            file_path: "src/main.py",
            sha256: "c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678",
            created_at: "2024-01-20T14:00:00Z"
          },
          status: "modified"
        },
        {
          file_path: "src/new_feature.py",
          old_version: undefined,
          new_version: {
            id: 4,
            project_id: 1,
            file_path: "src/new_feature.py",
            sha256: "d4e5f6789012345678901234567890abcdef1234567890abcdef123456789",
            created_at: "2024-01-18T09:00:00Z"
          },
          status: "added"
        }
      ];
      
      setComparisons(mockComparisons);
      setActiveTab('comparison');
    } catch (error) {
      message.error('Version comparison failed');
    } finally {
      setCompareLoading(false);
    }
  };

  const handleViewHistory = (filePath: string) => {
    // TODO: Show file history timeline
    message.info(`Viewing history for file ${filePath}`);
  };

  const handleViewFile = (version: FileVersion) => {
    // TODO: Show file content
    message.info(`Viewing content for version ${version.id}`);
  };

  const handleViewDiff = (comparison: VersionComparison) => {
    navigate('/versions/diff', { state: { comparison } });
  };

  useEffect(() => {
    fetchVersions();
  }, [selectedProject, selectedFile]);

  useEffect(() => {
    // TODO: Fetch projects list
    setProjects([
      { id: 1, name: "E-commerce Platform" },
      { id: 2, name: "Data Analytics Dashboard" }
    ]);
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <Card title="File Version Management">
        <div style={{ marginBottom: 16 }}>
          <Space wrap>
            <Select
              placeholder="Select Project"
              style={{ width: 200 }}
              value={selectedProject}
              onChange={setSelectedProject}
              allowClear
            >
              {projects.map(project => (
                <Select.Option key={project.id} value={project.id}>
                  {project.name}
                </Select.Option>
              ))}
            </Select>
            <Select
              placeholder="Filter Files"
              style={{ width: 200 }}
              value={selectedFile}
              onChange={setSelectedFile}
              allowClear
            >
              <Select.Option value="main">main.py</Select.Option>
              <Select.Option value="utils">utils.py</Select.Option>
              <Select.Option value="models">models.py</Select.Option>
            </Select>
            <RangePicker
                placeholder={['Start Time', 'End Time']}
                onChange={(dates, dateStrings) => {
                  if (dates) {
                    handleCompareVersions(dates as [Dayjs, Dayjs]);
                  }
                }}
              />
            <Button 
              type="primary" 
              icon={<BranchesOutlined />}
              loading={compareLoading}
              onClick={() => handleCompareVersions(null)}
            >
              Version Comparison
            </Button>
          </Space>
        </div>

        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={[
            {
              key: 'versions',
              label: 'Version List',
              children: (
                <Table
                  columns={versionColumns}
                  dataSource={versions}
                  rowKey="id"
                  loading={loading}
                  pagination={{
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total) => `Total ${total} records`,
                  }}
                />
              )
            },
            {
              key: 'comparison',
              label: 'Version Comparison',
              children: (
                <Table
                  columns={comparisonColumns}
                  dataSource={comparisons}
                  rowKey="file_path"
                  loading={compareLoading}
                  pagination={{
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: (total) => `Total ${total} file changes`,
                  }}
                />
              )
            },
            {
              key: 'timeline',
              label: 'Change Timeline',
              children: (
                <Timeline
                  items={versions.slice(0, 10).map(version => ({
                    children: (
                      <div>
                        <div><strong>{version.file_path}</strong></div>
                        <div style={{ color: '#666', fontSize: '12px' }}>
                          {new Date(version.created_at).toLocaleString()}
                        </div>
                        <div style={{ color: '#999', fontSize: '11px' }}>
                          SHA256: {version.sha256.substring(0, 16)}...
                        </div>
                      </div>
                    ),
                    color: 'blue'
                  }))}
                />
              )
            }
          ]}
        />
      </Card>

      <Card title="Example demonstrations" style={{ marginTop: 16 }}>
        <List
          dataSource={[
            'Compare the release branch against the baseline before approving a deployment.',
            'Open the diff viewer to review refactors with product and QA stakeholders.',
            'Use the timeline tab to brief leadership on change velocity each sprint.',
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

export default Versions;