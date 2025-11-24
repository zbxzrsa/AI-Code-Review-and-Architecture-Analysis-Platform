import React from 'react';

/**
 * 响应式与可访问性检查清单
 *
 * 此文件定义了最终检查应包含的项目
 */

export const RESPONSIVE_CHECKLIST = {
  breakpoints: {
    xs: 320,   // 小手机
    sm: 640,   // 大手机
    md: 1024,  // 平板
    lg: 1280,  // 桌面
    xl: 1536,  // 大屏幕
  },

  // 响应式检查项
  mobileAdaptation: [
    '✓ 按钮/触摸目标最小 48px',
    '✓ 侧边栏在移动端折叠为汉堡菜单',
    '✓ 表格使用卡片或堆叠布局（xs）',
    '✓ 文本可缩放 90%-110%',
    '✓ 无水平滚动（xs 视口）',
  ],

  // 可访问性检查项
  a11yChecklist: [
    '✓ 焦点样式清晰且对比度 > 3:1',
    '✓ 所有交互元素都能被键盘访问',
    '✓ 图标元素有 aria-label 或 title',
    '✓ 表单标签与输入关联（label htmlFor）',
    '✓ 颜色对比度 >= 4.5:1（正文）/ 3:1（大字体）',
    '✓ 页面有合理的标题层级（h1, h2, ...）',
    '✓ 用 Skip Link 跳过重复内容',
    '✓ 动画可被禁用（prefers-reduced-motion）',
  ],

  // 主题/设计一致性
  designChecklist: [
    '✓ 主色、成功、警告、错误、信息 5 色一致',
    '✓ Card 圆角 8px、阴影弱化、padding 24px',
    '✓ 按钮高度 40px，边框圆角 4px',
    '✓ 字体：基础 14px、标题 24px、代码 13px',
    '✓ 间距遵循 4/8/16/24/32/48 倍数',
    '✓ 暗模式与亮模式对比充分',
  ],

  // 性能
  performanceChecklist: [
    '✓ LCP < 2.5s（首屏内容最大元素）',
    '✓ FID < 100ms（首次输入延迟）',
    '✓ CLS < 0.1（布局抖动）',
    '✓ 大列表使用虚拟滚动',
    '✓ 图像使用 WebP + fallback',
    '✓ 代码分割关键路由',
  ],
};

/**
 * 响应式媒体查询辅助
 */
export const media = {
  xs: '@media (max-width: 640px)',
  sm: '@media (min-width: 640px) and (max-width: 1024px)',
  md: '@media (min-width: 1024px) and (max-width: 1280px)',
  lg: '@media (min-width: 1280px) and (max-width: 1536px)',
  xl: '@media (min-width: 1536px)',

  // 快捷版本
  mobile: '@media (max-width: 1024px)',
  desktop: '@media (min-width: 1024px)',
  tablet: '@media (min-width: 640px) and (max-width: 1024px)',

  // prefers-reduced-motion（无障碍）
  reducedMotion: '@media (prefers-reduced-motion: reduce)',
};

/**
 * 辅助函数：检测视口大小
 */
export function useResponsive() {
  const [screenSize, setScreenSize] = React.useState<'xs' | 'sm' | 'md' | 'lg' | 'xl'>('lg');

  React.useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width < 640) setScreenSize('xs');
      else if (width < 1024) setScreenSize('sm');
      else if (width < 1280) setScreenSize('md');
      else if (width < 1536) setScreenSize('lg');
      else setScreenSize('xl');
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return {
    screenSize,
    isMobile: screenSize === 'xs' || screenSize === 'sm',
    isTablet: screenSize === 'sm' || screenSize === 'md',
    isDesktop: screenSize === 'lg' || screenSize === 'xl',
  };
}
