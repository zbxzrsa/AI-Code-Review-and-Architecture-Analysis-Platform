import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import UserOnboarding, { OnboardingStep, shouldShowOnboarding } from './UserOnboarding';
import { Image, Typography, Card, Row, Col } from 'antd';
import { CodeOutlined, BranchesOutlined, BugOutlined, RocketOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

// 首次使用引导步骤
const firstTimeSteps: OnboardingStep[] = [
  {
    title: '欢迎使用智能代码审查与架构分析平台',
    description: '让我们快速了解平台的核心功能，帮助您提升代码质量和架构设计。',
    content: (
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Image
            src="/logo512.png"
            alt="平台概览"
            style={{ width: '100%', borderRadius: 8 }}
            preview={false}
          />
        </Col>
      </Row>
    ),
  },
  {
    title: '代码审查与质量分析',
    description: '自动识别代码中的潜在问题，提供智能修复建议，确保代码质量。',
    content: (
      <Card>
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <CodeOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          </Col>
          <Col span={18}>
            <Paragraph>
              • 自动检测代码异味和潜在问题<br />
              • 提供智能修复建议<br />
              • 代码质量指标可视化<br />
              • 支持多种编程语言
            </Paragraph>
          </Col>
        </Row>
      </Card>
    ),
  },
  {
    title: '架构分析与可视化',
    description: '直观展示代码架构，识别依赖关系，帮助优化系统设计。',
    content: (
      <Card>
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <BranchesOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          </Col>
          <Col span={18}>
            <Paragraph>
              • 代码架构可视化<br />
              • 依赖关系分析<br />
              • 架构异味检测<br />
              • 模块化建议
            </Paragraph>
          </Col>
        </Row>
      </Card>
    ),
  },
  {
    title: '开始您的第一个项目',
    description: '只需几个简单步骤，即可开始分析您的代码库。',
    content: (
      <Card>
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <RocketOutlined style={{ fontSize: 48, color: '#1677ff' }} />
          </Col>
          <Col span={18}>
            <Paragraph>
              1. 连接代码仓库或上传代码<br />
              2. 选择分析类型和范围<br />
              3. 启动分析<br />
              4. 查看分析结果和建议
            </Paragraph>
          </Col>
        </Row>
      </Card>
    ),
  },
];

// 功能引导步骤 - 代码分析页面
const codeAnalysisSteps: OnboardingStep[] = [
  {
    title: '代码分析功能',
    description: '了解如何使用代码分析功能检测潜在问题。',
    content: null,
    target: '.code-analysis-header',
    placement: 'bottom',
  },
  {
    title: '选择分析范围',
    description: '您可以选择特定文件或整个项目进行分析。',
    content: null,
    target: '.analysis-scope-selector',
    placement: 'right',
  },
  {
    title: '分析设置',
    description: '自定义分析规则和严重程度阈值。',
    content: null,
    target: '.analysis-settings-panel',
    placement: 'left',
  },
  {
    title: '启动分析',
    description: '点击此按钮开始代码分析。',
    content: null,
    target: '.start-analysis-button',
    placement: 'bottom',
  },
];

// 功能引导步骤 - 项目页面
const projectsSteps: OnboardingStep[] = [
  {
    title: '项目管理',
    description: '在这里管理您的所有项目。',
    content: null,
    target: '.projects-header',
    placement: 'bottom',
  },
  {
    title: '创建新项目',
    description: '点击此按钮创建新项目。',
    content: null,
    target: '.create-project-button',
    placement: 'right',
  },
  {
    title: '项目过滤',
    description: '使用这些选项过滤项目列表。',
    content: null,
    target: '.project-filters',
    placement: 'bottom',
  },
];

// 引导管理器组件
const OnboardingManager: React.FC = () => {
  const [showFirstTimeOnboarding, setShowFirstTimeOnboarding] = useState(false);
  const [showFeatureOnboarding, setShowFeatureOnboarding] = useState(false);
  const [currentSteps, setCurrentSteps] = useState<OnboardingStep[]>([]);
  const [currentTourId, setCurrentTourId] = useState('');
  const location = useLocation();
  
  // 检查是否应该显示引导
  useEffect(() => {
    // 首次使用引导
    const shouldShowFirstTime = shouldShowOnboarding('first-time');
    if (shouldShowFirstTime) {
      setShowFirstTimeOnboarding(true);
      setCurrentSteps(firstTimeSteps);
      setCurrentTourId('first-time');
      return;
    }
    
    // 基于路径的功能引导
    const path = location.pathname;
    
    if (path.includes('/code-analysis') && shouldShowOnboarding('code-analysis')) {
      setShowFeatureOnboarding(true);
      setCurrentSteps(codeAnalysisSteps);
      setCurrentTourId('code-analysis');
    } else if (path.includes('/projects') && shouldShowOnboarding('projects')) {
      setShowFeatureOnboarding(true);
      setCurrentSteps(projectsSteps);
      setCurrentTourId('projects');
    } else {
      setShowFeatureOnboarding(false);
    }
  }, [location.pathname]);
  
  // 完成首次使用引导
  const handleFirstTimeComplete = () => {
    setShowFirstTimeOnboarding(false);
  };
  
  // 跳过首次使用引导
  const handleFirstTimeSkip = () => {
    setShowFirstTimeOnboarding(false);
  };
  
  // 完成功能引导
  const handleFeatureComplete = () => {
    setShowFeatureOnboarding(false);
  };
  
  // 跳过功能引导
  const handleFeatureSkip = () => {
    setShowFeatureOnboarding(false);
  };
  
  return (
    <>
      {/* 首次使用引导 */}
      <UserOnboarding
        steps={firstTimeSteps}
        onComplete={handleFirstTimeComplete}
        onSkip={handleFirstTimeSkip}
        visible={showFirstTimeOnboarding}
        tourId="first-time"
      />
      
      {/* 功能引导 */}
      <UserOnboarding
        steps={currentSteps}
        onComplete={handleFeatureComplete}
        onSkip={handleFeatureSkip}
        visible={showFeatureOnboarding}
        tourId={currentTourId}
      />
    </>
  );
};

export default OnboardingManager;
