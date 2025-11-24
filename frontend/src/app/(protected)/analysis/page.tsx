/**
 * CodeAnalysis / 工作台
 *
 * 功能：
 * - 左：文件树（按严重度着色与数量徽章）
 * - 中：Monaco Editor，顶部工具条（运行、格式化、跳转、过滤），底部 ProblemsPanel
 * - 右：AIChatPanel（Summarize/Suggest tests/Explain diff）
 * - 行内装饰：根据 Findings 显示标记与 hover 提示
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Layout, Button, Space, Input, Tooltip, Tabs, Badge, Tree, Card, Drawer, Skeleton, Spin, Empty } from 'antd';
import {
  PlayCircleOutlined,
  FormatPainterOutlined,
  LinkOutlined,
  FilterOutlined,
  FolderOutlined,
  FileOutlined,
  BugOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import styled from 'styled-components';
import PageHeader from '../../../components/ui/PageHeader';
import SeverityTag from '../../../components/ui/SeverityTag';
import { spacing, borderRadius, shadows } from '../../../styles/tokens';

const { Sider, Content, Footer } = Layout;

// ============ 样式定义 ============
const LayoutWrapper = styled(Layout)`
  min-height: calc(100vh - 120px);
  display: flex;
`;

const EditorPanel = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  background: var(--bg-base);
  border: 1px solid var(--border-default);
  border-radius: ${borderRadius.sm}px;
`;

const Toolbar = styled.div`
  padding: ${spacing.md}px;
  border-bottom: 1px solid var(--border-default);
  display: flex;
  gap: ${spacing.md}px;
  align-items: center;
  background: var(--bg-secondary);
`;

const ProblemsPanel = styled.div`
  max-height: 200px;
  overflow-y: auto;
  border-top: 1px solid var(--border-default);
  padding: ${spacing.md}px;
  background: var(--bg-secondary);
`;

const ProblemRow = styled.div<{ severity: string }>`
  padding: ${spacing.sm}px;
  background: var(--bg-base);
  border-left: 3px solid ${(p) => {
    const colors: Record<string, string> = {
      CRITICAL: '#D92D20',
      HIGH: '#F04438',
      MEDIUM: '#F79009',
      LOW: '#12B76A',
      INFO: '#2E90FA',
    };
    return colors[p.severity] || '#64748B';
  }};
  margin-bottom: ${spacing.xs}px;
  cursor: pointer;
  &:hover {
    background: var(--bg-tertiary);
  }
`;

const FileTreeWrapper = styled.div`
  padding: ${spacing.md}px;
  flex: 1;
  overflow-y: auto;
`;

const AIChatWrapper = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: ${spacing.md}px;
  }
  .input {
    padding: ${spacing.md}px;
    border-top: 1px solid var(--border-default);
  }
`;

// ============ Mock 数据 ============
const MOCK_FILES = [
  { id: '1', name: 'src/', isDir: true, children: ['1-1', '1-2', '1-3'] },
  { id: '1-1', name: 'utils.ts', isDir: false, severity: 'MEDIUM', issues: 2 },
  { id: '1-2', name: 'api.ts', isDir: false, severity: 'HIGH', issues: 1 },
  { id: '1-3', name: 'index.ts', isDir: false, severity: 'LOW', issues: 0 },
  { id: '2', name: 'tests/', isDir: true, children: [] },
];

const MOCK_PROBLEMS = [
  { id: 'p1', severity: 'HIGH', line: 42, message: 'Potential null reference exception' },
  { id: 'p2', severity: 'MEDIUM', line: 87, message: 'Unused variable "tmp"' },
  { id: 'p3', severity: 'LOW', line: 120, message: 'Consider const instead of let' },
];

// ============ 文件树节点 ============
const buildTreeData = (files: any[]) => {
  return files
    .filter((f) => !f.id.includes('-') || f.id.split('-').length === 2) // 只显示一级
    .map((f) => ({
      key: f.id,
      title: (
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {f.isDir ? <FolderOutlined /> : <FileOutlined />}
          {f.name}
          {f.issues && f.issues > 0 && <Badge count={f.issues} style={{ backgroundColor: '#F04438' }} />}
        </span>
      ),
      children: f.children?.map((childId: string) => {
        const child = MOCK_FILES.find((c) => c.id === childId);
        return {
          key: child?.id,
          title: (
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {child?.isDir ? <FolderOutlined /> : <FileOutlined />}
              {child?.name}
              {child?.issues && child.issues > 0 && <Badge count={child.issues} style={{ backgroundColor: '#F04438' }} />}
            </span>
          ),
        };
      }),
    }));
};

// ============ 主组件 ============
const CodeAnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string | null>('1-1');
  const [editorContent, setEditorContent] = useState('// 示例代码\nfunction example() {\n  // ...\n}');
  const [showChatPanel, setShowChatPanel] = useState(false);
  const [expandedKeys, setExpandedKeys] = useState(['1']);
  const [problems, setProblems] = useState(MOCK_PROBLEMS);
  const [loading, setLoading] = useState(false);

  const handleRunAnalysis = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      // 模拟分析结果
      setProblems([
        ...MOCK_PROBLEMS,
        { id: 'p4', severity: 'MEDIUM', line: 150, message: 'Function too complex' },
      ]);
    }, 1500);
  };

  return (
    <div>
      <PageHeader
        title="Code Analysis Workbench"
        subtitle="Analyze and review your code with AI assistance"
        primaryAction={<Button type="primary" onClick={handleRunAnalysis} loading={loading} icon={<PlayCircleOutlined />}>Run Analysis</Button>}
      />

      <LayoutWrapper>
        {/* 左：文件树 */}
        <Sider width={240} style={{ background: 'var(--bg-secondary)', borderRight: '1px solid var(--border-default)' }}>
          <FileTreeWrapper>
            <Tree
              treeData={buildTreeData(MOCK_FILES)}
              expandedKeys={expandedKeys}
              onExpand={(keys) => setExpandedKeys(keys as string[])}
              onSelect={(keys) => setSelectedFile((keys[0] as string) || null)}
              selectedKeys={selectedFile ? [selectedFile] : []}
            />
          </FileTreeWrapper>
        </Sider>

        {/* 中：编辑器 + 问题面板 */}
        <Content style={{ padding: spacing.lg, display: 'flex', flexDirection: 'column' }}>
          <EditorPanel>
            {/* 工具条 */}
            <Toolbar>
              <Tooltip title="Run Analysis">
                <Button icon={<PlayCircleOutlined />} onClick={handleRunAnalysis} loading={loading} />
              </Tooltip>
              <Tooltip title="Format Code">
                <Button icon={<FormatPainterOutlined />} />
              </Tooltip>
              <Tooltip title="Go to Line">
                <Button icon={<LinkOutlined />} />
              </Tooltip>
              <Tooltip title="Filter Problems">
                <Button icon={<FilterOutlined />} />
              </Tooltip>
              <Input.Search placeholder="Search in file..." style={{ width: 200 }} />
            </Toolbar>

            {/* 编辑器区域 */}
            <div
              style={{
                flex: 1,
                padding: spacing.md,
                fontFamily: 'Monaco, monospace',
                fontSize: 13,
                overflow: 'auto',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                lineHeight: 1.6,
              }}
            >
              {loading ? <Spin /> : editorContent}
            </div>

            {/* 问题面板 */}
            <ProblemsPanel>
              <div style={{ fontWeight: 600, marginBottom: spacing.md }}>
                Problems ({problems.length})
              </div>
              {problems.length === 0 ? (
                <Empty description="No issues found" />
              ) : (
                problems.map((p) => (
                  <ProblemRow key={p.id} severity={p.severity}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>
                        {p.severity === 'HIGH' && <BugOutlined style={{ marginRight: 8, color: '#F04438' }} />}
                        {p.severity === 'MEDIUM' && <WarningOutlined style={{ marginRight: 8, color: '#F79009' }} />}
                        {p.severity === 'LOW' && <CheckCircleOutlined style={{ marginRight: 8, color: '#12B76A' }} />}
                        Line {p.line}: {p.message}
                      </span>
                      <SeverityTag severity={p.severity} />
                    </div>
                  </ProblemRow>
                ))
              )}
            </ProblemsPanel>
          </EditorPanel>
        </Content>

        {/* 右：AI 聊天面板 */}
        <Drawer
          title="AI Assistant"
          placement="right"
          onClose={() => setShowChatPanel(false)}
          open={showChatPanel}
          width={400}
        >
          <AIChatWrapper>
            <div className="messages">
              <Card>
                <p>AI 助手已就绪。你可以：</p>
                <ul>
                  <li>Summarize 此文件</li>
                  <li>Suggest unit tests</li>
                  <li>Explain the differences</li>
                </ul>
              </Card>
            </div>
            <div className="input">
              <Space.Compact style={{ width: '100%' }}>
                <Input placeholder="Ask AI..." />
                <Button type="primary">Send</Button>
              </Space.Compact>
            </div>
          </AIChatWrapper>
        </Drawer>

        {/* AI 按钮 */}
        <div style={{ position: 'absolute', bottom: 20, right: 20 }}>
          <Tooltip title="AI Assistant">
            <Button
              type="primary"
              shape="circle"
              size="large"
              icon={<RobotOutlined />}
              onClick={() => setShowChatPanel(true)}
            />
          </Tooltip>
        </div>
      </LayoutWrapper>
    </div>
  );
};

export default CodeAnalysisPage;
