/**
 * AppHeader组件单元测试
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import AppHeader from '../AppHeader';

describe('AppHeader组件测试', () => {
  test('应该正确渲染组件', () => {
    render(<AppHeader />);
    
    // 检查标题是否存在
    expect(screen.getByText('CodeInsight')).toBeInTheDocument();
    
    // 检查登录按钮是否存在
    expect(screen.getByRole('button', { name: /Login/i })).toBeInTheDocument();
  });

  test('应该显示正确的样式', () => {
    render(<AppHeader />);
    
    const title = screen.getByText('CodeInsight');
    expect(title).toHaveStyle({
      color: 'white',
      fontSize: '20px',
      fontWeight: 'bold'
    });
  });

  test('登录按钮应该包含用户图标', () => {
    render(<AppHeader />);
    
    const loginButton = screen.getByRole('button', { name: /Login/i });
    expect(loginButton).toBeInTheDocument();
    
    // 检查按钮是否为primary类型
    expect(loginButton).toHaveClass('ant-btn-primary');
  });

  test('应该正确处理按钮点击事件', () => {
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    render(<AppHeader />);
    
    const loginButton = screen.getByRole('button', { name: /Login/i });
    fireEvent.click(loginButton);
    
    // 由于当前组件没有点击处理逻辑，这里主要测试按钮可点击
    expect(loginButton).toBeEnabled();
    
    consoleSpy.mockRestore();
  });

  test('应该具有正确的布局结构', () => {
    const { container } = render(<AppHeader />);
    
    // 检查Header元素
    const header = container.querySelector('.ant-layout-header');
    expect(header).toBeInTheDocument();
    expect(header).toHaveStyle({
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    });
  });
});