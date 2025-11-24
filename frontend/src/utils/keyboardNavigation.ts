/**
 * 键盘导航辅助工具
 * 提供全局键盘导航支持，包括焦点管理、快捷键处理和无障碍增强
 */

import { useEffect, useRef, useState } from 'react';

// 焦点管理钩子
export const useFocusManagement = (elementRef: React.RefObject<HTMLElement>, options = { autoFocus: false }) => {
  useEffect(() => {
    if (options.autoFocus && elementRef.current) {
      elementRef.current.focus();
    }
    
    return () => {
      // 清理工作
    };
  }, [elementRef, options.autoFocus]);

  const focusElement = () => {
    if (elementRef.current) {
      elementRef.current.focus();
    }
  };

  return { focusElement };
};

// 快捷键处理钩子
export const useKeyboardShortcut = (
  targetKey: string | string[],
  callback: (event: KeyboardEvent) => void,
  options = { ctrl: false, alt: false, shift: false, meta: false, preventDefault: true }
) => {
  useEffect(() => {
    const keys = Array.isArray(targetKey) ? targetKey : [targetKey];
    
    const handleKeyDown = (event: KeyboardEvent) => {
      const keyMatches = keys.includes(event.key);
      const modifiersMatch = 
        (options.ctrl ? event.ctrlKey : !event.ctrlKey) &&
        (options.alt ? event.altKey : !event.altKey) &&
        (options.shift ? event.shiftKey : !event.shiftKey) &&
        (options.meta ? event.metaKey : !event.metaKey);
      
      if (keyMatches && modifiersMatch) {
        if (options.preventDefault) {
          event.preventDefault();
        }
        callback(event);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [targetKey, callback, options]);
};

// 焦点陷阱钩子 - 用于模态框和弹出菜单
export const useFocusTrap = (containerRef: React.RefObject<HTMLElement>) => {
  const [isActive, setIsActive] = useState(false);
  
  useEffect(() => {
    if (!isActive || !containerRef.current) return;
    
    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      
      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastElement) {
          firstElement.focus();
          e.preventDefault();
        }
      }
    };
    
    document.addEventListener('keydown', handleTabKey);
    
    // 保存当前焦点并将焦点移至容器
    const previouslyFocused = document.activeElement as HTMLElement;
    firstElement?.focus();
    
    return () => {
      document.removeEventListener('keydown', handleTabKey);
      previouslyFocused?.focus();
    };
  }, [isActive, containerRef]);
  
  return { activate: () => setIsActive(true), deactivate: () => setIsActive(false) };
};

// 快捷键帮助对话框状态
let isHelpDialogOpen = false;

// 注册全局快捷键
export const registerGlobalShortcuts = () => {
  // 帮助对话框快捷键 (?)
  document.addEventListener('keydown', (e) => {
    if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
      e.preventDefault();
      if (!isHelpDialogOpen) {
        showKeyboardShortcutsHelp();
      }
    }
    
    // ESC键关闭帮助对话框
    if (e.key === 'Escape' && isHelpDialogOpen) {
      hideKeyboardShortcutsHelp();
    }
  });
};

// 显示键盘快捷键帮助
const showKeyboardShortcutsHelp = () => {
  isHelpDialogOpen = true;
  
  // 创建帮助对话框
  const helpDialog = document.createElement('div');
  helpDialog.id = 'keyboard-shortcuts-help';
  helpDialog.setAttribute('role', 'dialog');
  helpDialog.setAttribute('aria-modal', 'true');
  helpDialog.setAttribute('aria-labelledby', 'keyboard-shortcuts-title');
  
  helpDialog.style.position = 'fixed';
  helpDialog.style.top = '50%';
  helpDialog.style.left = '50%';
  helpDialog.style.transform = 'translate(-50%, -50%)';
  helpDialog.style.backgroundColor = '#fff';
  helpDialog.style.padding = '20px';
  helpDialog.style.borderRadius = '8px';
  helpDialog.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
  helpDialog.style.zIndex = '9999';
  helpDialog.style.maxWidth = '600px';
  helpDialog.style.width = '90%';
  
  // 添加标题和内容
  helpDialog.innerHTML = `
    <h2 id="keyboard-shortcuts-title" style="margin-top: 0;">键盘快捷键</h2>
    <div style="max-height: 400px; overflow-y: auto;">
      <table style="width: 100%; border-collapse: collapse;">
        <thead>
          <tr>
            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #eee;">快捷键</th>
            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #eee;">功能</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>?</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">显示此帮助</td>
          </tr>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>Esc</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">关闭弹窗或取消操作</td>
          </tr>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>/</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">聚焦到搜索框</td>
          </tr>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>Ctrl</kbd> + <kbd>Enter</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">提交当前表单</td>
          </tr>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>Tab</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">移动到下一个可聚焦元素</td>
          </tr>
          <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><kbd>Shift</kbd> + <kbd>Tab</kbd></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">移动到上一个可聚焦元素</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top: 20px; text-align: right;">
      <button id="close-shortcuts-help" style="padding: 8px 16px; background-color: #1890ff; color: white; border: none; border-radius: 4px; cursor: pointer;">关闭</button>
    </div>
  `;
  
  // 添加到文档
  document.body.appendChild(helpDialog);
  
  // 添加关闭按钮事件
  const closeButton = document.getElementById('close-shortcuts-help');
  if (closeButton) {
    closeButton.focus();
    closeButton.addEventListener('click', hideKeyboardShortcutsHelp);
  }
  
  // 添加背景遮罩
  const overlay = document.createElement('div');
  overlay.id = 'keyboard-shortcuts-overlay';
  overlay.style.position = 'fixed';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.width = '100%';
  overlay.style.height = '100%';
  overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  overlay.style.zIndex = '9998';
  
  overlay.addEventListener('click', hideKeyboardShortcutsHelp);
  document.body.appendChild(overlay);
};

// 隐藏键盘快捷键帮助
const hideKeyboardShortcutsHelp = () => {
  const helpDialog = document.getElementById('keyboard-shortcuts-help');
  const overlay = document.getElementById('keyboard-shortcuts-overlay');
  
  if (helpDialog) {
    document.body.removeChild(helpDialog);
  }
  
  if (overlay) {
    document.body.removeChild(overlay);
  }
  
  isHelpDialogOpen = false;
};

// 初始化键盘导航
export const initKeyboardNavigation = () => {
  registerGlobalShortcuts();
  
  // 为搜索框添加快捷键
  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && !e.ctrlKey && !e.altKey && !e.metaKey && 
        !(document.activeElement instanceof HTMLInputElement) && 
        !(document.activeElement instanceof HTMLTextAreaElement)) {
      e.preventDefault();
      
      // 查找搜索框并聚焦
      const searchInput = document.querySelector('input[type="search"], input[placeholder*="搜索"], input[placeholder*="search"]') as HTMLElement;
      if (searchInput) {
        searchInput.focus();
      }
    }
  });
};

// 导出默认初始化函数
export default initKeyboardNavigation;