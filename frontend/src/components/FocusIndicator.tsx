import React, { useEffect } from 'react';

/**
 * 焦点指示器组件
 * 为键盘导航提供可视化焦点指示
 */
const FocusIndicator: React.FC = () => {
  useEffect(() => {
    // 添加全局样式类以增强键盘焦点可见性
    const style = document.createElement('style');
    style.innerHTML = `
      :focus-visible {
        outline: 2px solid #1890ff !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2) !important;
        transition: outline-offset 0.1s ease !important;
      }
      
      /* 为不同类型的元素提供特定的焦点样式 */
      button:focus-visible, 
      [role="button"]:focus-visible {
        outline-offset: 4px !important;
      }
      
      input:focus-visible, 
      textarea:focus-visible, 
      select:focus-visible {
        border-color: #1890ff !important;
      }
      
      a:focus-visible {
        text-decoration: underline !important;
      }
      
      /* 添加焦点动画 */
      .focus-animation {
        position: absolute;
        border-radius: 3px;
        pointer-events: none;
        z-index: 9999;
        box-shadow: 0 0 0 2px #1890ff;
        transition: all 0.2s ease-out;
        opacity: 0;
      }
    `;
    document.head.appendChild(style);

    // 创建焦点动画元素
    const focusAnimation = document.createElement('div');
    focusAnimation.className = 'focus-animation';
    document.body.appendChild(focusAnimation);

    // 监听焦点变化
    const handleFocusIn = (e: FocusEvent) => {
      const target = e.target as HTMLElement;
      
      // 仅对键盘导航触发的焦点变化做动画
      if (target && target.tagName && !target.classList.contains('ant-select-dropdown')) {
        const rect = target.getBoundingClientRect();
        
        focusAnimation.style.width = `${rect.width + 4}px`;
        focusAnimation.style.height = `${rect.height + 4}px`;
        focusAnimation.style.left = `${window.scrollX + rect.left - 2}px`;
        focusAnimation.style.top = `${window.scrollY + rect.top - 2}px`;
        focusAnimation.style.opacity = '1';
        
        // 动画结束后淡出
        setTimeout(() => {
          focusAnimation.style.opacity = '0';
        }, 600);
      }
    };

    document.addEventListener('focusin', handleFocusIn);

    // 清理函数
    return () => {
      document.removeEventListener('focusin', handleFocusIn);
      if (document.body.contains(focusAnimation)) {
        document.body.removeChild(focusAnimation);
      }
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, []);

  return null; // 这是一个无UI组件，只添加功能
};

export default FocusIndicator;