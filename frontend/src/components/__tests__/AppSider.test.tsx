/**
 * AppSider组件单元测试
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import AppSider from '../AppSider';

// Mock react-router-dom
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' })
}));

// 包装组件以提供Router上下文
const renderWithRouter = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('AppSider组件测试', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  test('应该正确渲染所有菜单项', () => {
    renderWithRouter(<AppSider />);
    
    // 检查所有菜单项
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Code Analysis')).toBeInTheDocument();
    expect(screen.getByText('Projects')).toBeInTheDocument();
  });

  test('应该显示正确的图标', () => {
    renderWithRouter(<AppSider />);
    
    // 检查图标是否存在（通过class检查）
    const container = screen.getByText('Dashboard').closest('.ant-menu-item');
    expect(container?.querySelector('.anticon-dashboard')).toBeInTheDocument();
    
    const codeContainer = screen.getByText('Code Analysis').closest('.ant-menu-item');
    expect(codeContainer?.querySelector('.anticon-code')).toBeInTheDocument();
    
    const projectContainer = screen.getByText('Projects').closest('.ant-menu-item');
    expect(projectContainer?.querySelector('.anticon-project')).toBeInTheDocument();
  });

  test('应该正确处理菜单项点击', () => {
    renderWithRouter(<AppSider />);
    
    // 点击代码分析菜单项
    fireEvent.click(screen.getByText('Code Analysis'));
    expect(mockNavigate).toHaveBeenCalledWith('/analysis');
    
    // 点击项目管理菜单项
    fireEvent.click(screen.getByText('Projects'));
    expect(mockNavigate).toHaveBeenCalledWith('/projects');
  });

  test('应该支持折叠功能', () => {
    const { container } = renderWithRouter(<AppSider />);
    
    // 查找折叠按钮
    const collapseButton = container.querySelector('.ant-layout-sider-trigger');
    expect(collapseButton).toBeInTheDocument();
    
    // 点击折叠按钮
    if (collapseButton) {
      fireEvent.click(collapseButton);
      
      // 检查是否添加了折叠类
      const sider = container.querySelector('.ant-layout-sider');
      expect(sider).toHaveClass('ant-layout-sider-collapsed');
    }
  });

  test('应该根据当前路径高亮对应菜单项', () => {
    // Mock不同的路径
    jest.mocked(require('react-router-dom').useLocation).mockReturnValue({ pathname: '/analysis' });
    
    renderWithRouter(<AppSider />);
    
    // 检查代码分析菜单项是否被选中
    const analysisItem = screen.getByText('Code Analysis').closest('.ant-menu-item');
    expect(analysisItem).toHaveClass('ant-menu-item-selected');
  });

  test('应该使用深色主题', () => {
    const { container } = renderWithRouter(<AppSider />);
    
    const menu = container.querySelector('.ant-menu');
    expect(menu).toHaveClass('ant-menu-dark');
  });

  test('应该是内联模式', () => {
    const { container } = renderWithRouter(<AppSider />);
    
    const menu = container.querySelector('.ant-menu');
    expect(menu).toHaveClass('ant-menu-inline');
  });
});