import React, { useState, useEffect } from 'react';
import { Modal, Steps, Button, Typography, Space, Card, Progress, Tooltip } from 'antd';
import { CheckOutlined, RightOutlined, CloseOutlined } from '@ant-design/icons';
import { motion } from 'framer-motion';

const { Title, Paragraph, Text } = Typography;
const { Step } = Steps;

// 引导步骤类型
export interface OnboardingStep {
  title: string;
  description: string;
  content: React.ReactNode;
  target?: string; // 目标元素选择器
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

// 引导系统属性
interface UserOnboardingProps {
  steps: OnboardingStep[];
  onComplete: () => void;
  onSkip?: () => void;
  visible: boolean;
  tourId: string; // 用于区分不同的引导流程
}

// 用户引导组件
const UserOnboarding: React.FC<UserOnboardingProps> = ({
  steps,
  onComplete,
  onSkip,
  visible,
  tourId,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetElement, setTargetElement] = useState<HTMLElement | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  
  // 获取目标元素并计算提示框位置
  useEffect(() => {
    if (visible && steps[currentStep]?.target) {
      const element = document.querySelector(steps[currentStep].target!) as HTMLElement;
      if (element) {
        setTargetElement(element);
        
        const rect = element.getBoundingClientRect();
        const placement = steps[currentStep].placement || 'bottom';
        
        let position = { top: 0, left: 0 };
        
        switch (placement) {
          case 'top':
            position = {
              top: rect.top - 10,
              left: rect.left + rect.width / 2,
            };
            break;
          case 'bottom':
            position = {
              top: rect.bottom + 10,
              left: rect.left + rect.width / 2,
            };
            break;
          case 'left':
            position = {
              top: rect.top + rect.height / 2,
              left: rect.left - 10,
            };
            break;
          case 'right':
            position = {
              top: rect.top + rect.height / 2,
              left: rect.right + 10,
            };
            break;
        }
        
        setTooltipPosition(position);
      }
    }
  }, [visible, currentStep, steps]);
  
  // 下一步
  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };
  
  // 上一步
  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };
  
  // 完成引导
  const handleComplete = () => {
    // 保存引导完成状态
    localStorage.setItem(`onboarding_${tourId}_completed`, 'true');
    onComplete();
  };
  
  // 跳过引导
  const handleSkip = () => {
    // 保存引导跳过状态
    localStorage.setItem(`onboarding_${tourId}_skipped`, 'true');
    onSkip?.();
  };
  
  // 如果没有目标元素，使用模态框展示
  if (!steps[currentStep]?.target) {
    return (
      <Modal
        title={<Title level={4}>欢迎使用智能代码审查与架构分析平台</Title>}
        open={visible}
        footer={null}
        closable={false}
        width={700}
        centered
      >
        <Steps current={currentStep} size="small" style={{ marginBottom: 24 }}>
          {steps.map((step) => (
            <Step key={step.title} title={step.title} />
          ))}
        </Steps>
        
        <Card style={{ marginBottom: 24 }}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Title level={4}>{steps[currentStep]?.title}</Title>
            <Paragraph>{steps[currentStep]?.description}</Paragraph>
            {steps[currentStep]?.content}
          </Space>
        </Card>
        
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button onClick={handleSkip}>跳过引导</Button>
          <Space>
            {currentStep > 0 && (
              <Button onClick={handlePrev}>上一步</Button>
            )}
            <Button type="primary" onClick={handleNext}>
              {currentStep < steps.length - 1 ? '下一步' : '完成'}
            </Button>
          </Space>
        </div>
      </Modal>
    );
  }
  
  // 使用工具提示展示目标元素引导
  return (
    <>
      {visible && targetElement && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 1000,
            pointerEvents: 'none',
          }}
        >
          {/* 高亮目标元素 */}
          <div
            style={{
              position: 'absolute',
              top: targetElement.getBoundingClientRect().top,
              left: targetElement.getBoundingClientRect().left,
              width: targetElement.offsetWidth,
              height: targetElement.offsetHeight,
              backgroundColor: 'transparent',
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5)',
              borderRadius: '4px',
              zIndex: 1001,
            }}
          />
          
          {/* 提示框 */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              position: 'absolute',
              top: tooltipPosition.top,
              left: tooltipPosition.left,
              transform: 'translate(-50%, -50%)',
              backgroundColor: 'white',
              padding: '16px',
              borderRadius: '8px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
              zIndex: 1002,
              width: '300px',
              pointerEvents: 'auto',
            }}
          >
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Title level={5} style={{ margin: 0 }}>{steps[currentStep].title}</Title>
                <Button
                  type="text"
                  icon={<CloseOutlined />}
                  size="small"
                  onClick={handleSkip}
                  aria-label="关闭引导"
                />
              </div>
              
              <Paragraph>{steps[currentStep].description}</Paragraph>
              
              <Progress
                percent={Math.round(((currentStep + 1) / steps.length) * 100)}
                size="small"
                showInfo={false}
                style={{ marginBottom: 8 }}
              />
              
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Space>
                  <Text type="secondary">
                    {currentStep + 1}/{steps.length}
                  </Text>
                </Space>
                <Space>
                  {currentStep > 0 && (
                    <Button size="small" onClick={handlePrev}>上一步</Button>
                  )}
                  <Button
                    type="primary"
                    size="small"
                    onClick={handleNext}
                    icon={currentStep < steps.length - 1 ? <RightOutlined /> : <CheckOutlined />}
                  >
                    {currentStep < steps.length - 1 ? '下一步' : '完成'}
                  </Button>
                </Space>
              </div>
            </Space>
          </motion.div>
        </div>
      )}
    </>
  );
};

// 检查是否应该显示引导
export const shouldShowOnboarding = (tourId: string): boolean => {
  const completed = localStorage.getItem(`onboarding_${tourId}_completed`) === 'true';
  const skipped = localStorage.getItem(`onboarding_${tourId}_skipped`) === 'true';
  return !completed && !skipped;
};

// 重置引导状态
export const resetOnboarding = (tourId: string): void => {
  localStorage.removeItem(`onboarding_${tourId}_completed`);
  localStorage.removeItem(`onboarding_${tourId}_skipped`);
};

export default UserOnboarding;