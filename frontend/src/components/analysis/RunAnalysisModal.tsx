import React, { useState } from 'react';
import { Modal, Form, Select, Button, Space, Typography, Alert, Steps } from 'antd';
import { 
  PlayCircleOutlined, 
  GithubOutlined, 
  FileSearchOutlined,
  SettingOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;
const { Step } = Steps;
const { Option } = Select;

interface RunAnalysisModalProps {
  visible: boolean;
  onClose: () => void;
}

const RunAnalysisModal: React.FC<RunAnalysisModalProps> = ({ visible, onClose }) => {
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [analysisType, setAnalysisType] = useState<string>('');
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const projects = [
    { id: '1', name: 'frontend-app', language: 'TypeScript/React' },
    { id: '2', name: 'backend-api', language: 'Python/FastAPI' },
    { id: '3', name: 'mobile-app', language: 'React Native' },
    { id: '4', name: 'data-pipeline', language: 'Python/Airflow' }
  ];

  const analysisTypes = [
    { 
      value: 'comprehensive', 
      label: 'Comprehensive Analysis',
      description: 'Full code review including security, performance, and maintainability'
    },
    { 
      value: 'security', 
      label: 'Security Focus',
      description: 'Deep dive into security vulnerabilities and best practices'
    },
    { 
      value: 'performance', 
      label: 'Performance Analysis',
      description: 'Identify performance bottlenecks and optimization opportunities'
    },
    { 
      value: 'architecture', 
      label: 'Architecture Review',
      description: 'Evaluate code structure, patterns, and design decisions'
    }
  ];

  const handleNext = async () => {
    if (currentStep === 0) {
      try {
        await form.validateFields(['project']);
        setSelectedProject(form.getFieldValue('project'));
        setCurrentStep(1);
      } catch (error) {
        // Form validation failed
      }
    } else if (currentStep === 1) {
      try {
        await form.validateFields(['analysisType']);
        setAnalysisType(form.getFieldValue('analysisType'));
        setCurrentStep(2);
      } catch (error) {
        // Form validation failed
      }
    } else if (currentStep === 2) {
      await runAnalysis();
    }
  };

  const handlePrev = () => {
    setCurrentStep(currentStep - 1);
  };

  const runAnalysis = async () => {
    setLoading(true);
    
    // Simulate analysis process
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    setAnalysisComplete(true);
    setLoading(false);
    
    // Auto-close after showing success
    setTimeout(() => {
      handleClose();
    }, 2000);
  };

  const handleClose = () => {
    form.resetFields();
    setCurrentStep(0);
    setAnalysisComplete(false);
    setSelectedProject('');
    setAnalysisType('');
    onClose();
  };

  const getStepIcon = (step: number) => {
    switch (step) {
      case 0: return <FileSearchOutlined />;
      case 1: return <SettingOutlined />;
      case 2: return <PlayCircleOutlined />;
      default: return null;
    }
  };

  return (
    <Modal
      title={
        <Space>
          <PlayCircleOutlined />
          Run Code Analysis
        </Space>
      }
      open={visible}
      onCancel={handleClose}
      width={600}
      footer={
        <Space>
          {currentStep > 0 && currentStep < 3 && (
            <Button onClick={handlePrev}>
              Previous
            </Button>
          )}
          {currentStep < 2 && (
            <Button type="primary" onClick={handleNext}>
              Next
            </Button>
          )}
          {currentStep === 2 && !analysisComplete && (
            <Button 
              type="primary" 
              onClick={handleNext}
              loading={loading}
              icon={<PlayCircleOutlined />}
            >
              Run Analysis
            </Button>
          )}
        </Space>
      }
    >
      <Steps current={currentStep} style={{ marginBottom: 24 }}>
        <Step title="Select Project" icon={getStepIcon(0)} />
        <Step title="Choose Analysis" icon={getStepIcon(1)} />
        <Step title="Execute" icon={getStepIcon(2)} />
      </Steps>

      {analysisComplete ? (
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <CheckCircleOutlined style={{ fontSize: '64px', color: '#52c41a' }} />
          <Title level={4} style={{ marginTop: 16 }}>
            Analysis Complete!
          </Title>
          <Text type="secondary">
            Redirecting to results...
          </Text>
        </div>
      ) : (
        <Form form={form} layout="vertical">
          {currentStep === 0 && (
            <div>
              <Title level={5}>Select Project to Analyze</Title>
              <Form.Item
                name="project"
                label="Project"
                rules={[{ required: true, message: 'Please select a project' }]}
              >
                <Select
                  placeholder="Choose a project"
                  size="large"
                  onChange={(value) => {
                    const project = projects.find(p => p.id === value);
                    setSelectedProject(project?.name || '');
                  }}
                >
                  {projects.map(project => (
                    <Option key={project.id} value={project.id}>
                      <Space>
                        <GithubOutlined />
                        <div>
                          <div>{project.name}</div>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {project.language}
                          </Text>
                        </div>
                      </Space>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
              
              {selectedProject && (
                <Alert
                  message={`Selected: ${selectedProject}`}
                  type="info"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </div>
          )}

          {currentStep === 1 && (
            <div>
              <Title level={5}>Choose Analysis Type</Title>
              <Form.Item
                name="analysisType"
                label="Analysis Type"
                rules={[{ required: true, message: 'Please select an analysis type' }]}
              >
                <Select
                  placeholder="Select analysis type"
                  size="large"
                  onChange={(value) => {
                    const type = analysisTypes.find(t => t.value === value);
                    setAnalysisType(type?.label || '');
                  }}
                >
                  {analysisTypes.map(type => (
                    <Option key={type.value} value={type.value}>
                      <div>
                        <div>{type.label}</div>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {type.description}
                        </Text>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
              
              {analysisType && (
                <Alert
                  message={`Selected: ${analysisType}`}
                  type="info"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </div>
          )}

          {currentStep === 2 && (
            <div>
              <Title level={5}>Ready to Run Analysis</Title>
              <Alert
                message="Analysis Configuration"
                description={
                  <div>
                    <p><strong>Project:</strong> {selectedProject}</p>
                    <p><strong>Analysis Type:</strong> {analysisType}</p>
                    <p><strong>Estimated Time:</strong> 2-5 minutes</p>
                  </div>
                }
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              <Alert
                message="What happens next?"
                description={
                  <ul>
                    <li>Code will be scanned and analyzed</li>
                    <li>AI will identify issues and suggestions</li>
                    <li>Results will be available in the analysis dashboard</li>
                    <li>You'll receive a notification when complete</li>
                  </ul>
                }
                type="info"
                showIcon
              />
            </div>
          )}
        </Form>
      )}
    </Modal>
  );
};

export default RunAnalysisModal;